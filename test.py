import numpy as np
import torch
import random

import PIL
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append("../../")

# seeding only for debugging
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Dataset loading
from data import dataloading
from data.era5_temp_dataset import InverseMinMaxScaler

from os.path import exists, join
import matplotlib.pyplot as plt
from matplotlib import transforms
import argparse
import timeit
import pdb

from models.architectures import condNF, srflow
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage

parser = argparse.ArgumentParser()

# train configs
parser.add_argument("--modeltype", type=str, default="flow",
                        help="Specify modeltype you would like to train [flow, diff, unet3d, convLSTM].")
parser.add_argument("--model_path", type=str, default="runs/",
                        help="Directory where models are saved.")
parser.add_argument("--modelname", type=str, default=None,
                        help="Sepcify modelname to be tested.")
parser.add_argument("--epochs", type=int, default=10000,
                        help="number of epochs")
parser.add_argument("--max_steps", type=int, default=2000000,
                        help="For training on a large dataset.")
parser.add_argument("--log_interval", type=int, default=100,
                        help="Interval in which results should be loggeexperiments/d.")
parser.add_argument("--val_interval", type=int, default=250,
                        help="Interval in which model should be validated.")

# runtime configs
parser.add_argument("--visual", action="store_true",
                        help="Visualizing the samples at test time.")
parser.add_argument("--noscaletest", action="store_true",
                        help="Disable scale in coupling layers only at test time.")
parser.add_argument("--noscale", action="store_true",
                        help="Disable scale in coupling layers.")
parser.add_argument("--testmode", action="store_true",
                        help="Model run on test set.")
parser.add_argument("--train", action="store_true",
                        help="If model should be trained.")
parser.add_argument("--resume", action="store_true",
                        help="If training should be resumed.")
parser.add_argument("--ds", action="store_true",
                    help="Applies downscaling as well.")

# hyperparameters
parser.add_argument("--nbits", type=int, default=8,
                        help="Images converted to n-bit representations.")
parser.add_argument("--s", type=int, default=1, help="Upscaling factor.")
parser.add_argument("--crop_size", type=int, default=500,
                        help="Crop size when random cropping is applied.")
parser.add_argument("--patch_size", type=int, default=500,
                        help="Training patch size.")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--seq_len", type=int, default=2,
                        help="Total sequnece length needed from dataloader")
parser.add_argument("--lag_len", type=int, default=2,
                        help="Lag length of time-series")
parser.add_argument("--lead_len", type=int, default=1,
                        help="Lead time length of time-series")
parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
parser.add_argument("--filter_size", type=int, default=512//2,
                        help="filter size NN in Affine Coupling Layer")
parser.add_argument("--Lst", type=int, default=3, help="# of levels")
parser.add_argument("--Kst", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
parser.add_argument("--Lsr", type=int, default=3, help="# of levels")
parser.add_argument("--Ksr", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128//8,
                        help="# of residual-in-residual blocks in LR network.")

# data
parser.add_argument("--datadir", type=str, default="/home/christina/Documents/climsim_ds/data",
                        help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="temp",
                        help="Dataset to train the model on.")

args = parser.parse_args()

def inv_scaler(x, min_value=0, max_value=100):
    x = x * (max_value - min_value) + min_value
    return x

def create_rollout(model, x_for, x_past, lead_time):
    """
    Generate a rollout sequence using the given forecasting model.

    Parameters:
    - model: The forecasting model.
    - x_for: The target future sequence.
    - x_past: The historical input sequence.
    - lead_time: The length of the lead time for the rollout.

    Returns:
    - stacked_pred: The stacked prediction sequence.
    - abs_err: The absolute error between predictions and the target future sequence.
    - nll: List of negative log likelihood values for each prediction step.
    """

    # Initialize predictions, negative log likelihood list, and initial state
    predictions = []
    nll = []

    # Obtain the initial prediction, state, and ignore third output
    past = x_past[0, :, :, :, :].unsqueeze(0)
    init_pred, s, _ = stmodel._predict(x_past=past, state=None, eps=0.8)

    # Append the initial prediction to the sequence
    predictions.append(init_pred[0, :, :, :, :])

    # Initialize intermediate state with the second time step of the historical input sequence
    interm = x_past[0, :, 1, :, :].unsqueeze(1).cuda()

    # Generate the rollout sequence
    for l in range(lead_time):

        # Concatenate the previous prediction and intermediate state
        context = torch.cat((predictions[l - 1], interm), 1)

        # Run the model to predict the next time step
        x, s, curr_nll = model._predict(x_past=context.unsqueeze(0), state=s)

        # Append the predicted time step and associated negative log likelihood to the lists
        predictions.append(x[0, :, :, :, :])
        nll.append(curr_nll.item())
        # print(curr_nll.item())
        curr_nll = 0
        # Update the intermediate state
        interm = x[0, :, :, :, :]

    # Stack the predictions to form the final sequence
    stacked_pred = torch.stack(predictions, dim=0).squeeze(1).squeeze(2)

    # Compute absolute error images
    abs_err = torch.abs(stacked_pred.cuda() - x_for[:, ...].cuda().squeeze(1))

    return stacked_pred, abs_err, nll


def test(model, test_loader, exp_name, modelname, logstep, args):

    state=None
    nll_list=[]
    mae08 = []
    rmse08 = []
    rmse08unorm = []
    mae08unorm = []
    avrg_fwd_time = []
    avrg_bw_time = []
    model.eval()
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    savedir = "experiments/{}_{}_{}_nods/snapshots/test_set".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir, exist_ok=True)
    savedir_txt = 'experiments/{}_{}_{}_nods/'.format(exp_name, modelname, args.trainset)
    os.makedirs(savedir_txt, exist_ok=True)

    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0].to(args.device)
            x_unorm = item[1].to(args.device)

            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]
            x_past_unorm, x_for_unorm = x_unorm[:,:2,...], x_unorm[:,2:,...]

            x_resh = F.interpolate(x[:,0,...], (x_for.shape[3]//args.s, x_for.shape[4]//args.s))

            start = timeit.default_timer()

            # split time series into lags and prediction window
            x_past_lr, x_for_lr = x_resh[:,:2,...], x_resh[:,2:,...]

            # reshape into correct format [bsz, num_channels, seq_len, height, width]
            x_past_lr = x_past_lr.unsqueeze(1).contiguous().float()
            x_for_lr = x_for_lr.unsqueeze(1).contiguous().float()

            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)

            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)
            avrg_fwd_time.append(stop-start)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            # ---------------------- Evaluate Predictions---------------------- #

            # Evaluate for different temperatures
            # mu0 = model._predict(x_past, state, eps=0)
            # mu05 = model._predict(x_past, state, eps=0.5)
            # mu08 = model._predict(x_past, state, eps=0.8)
            # mu1 = model._predict(x_past, state, eps=1)

            print(" Evaluate Predictions ... ")
            rollout_len = args.bsz - 1
            eps = 0.8
            predictions = []

            start = timeit.default_timer()
            past = x_past[0,:,:,:,:].unsqueeze(0)
            x, s = model._predict(x_past=past,
                                  state=None, eps=eps)

            stop = timeit.default_timer()

            print("Time Bwd pass / predicting:", stop - start)
            avrg_bw_time.append(stop - start)

            # create multiple rollouts with same initial conditions
            nr_of_rollouts = 4
            stacked_pred1, abs_err1 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred2, abs_err2 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred3, abs_err3 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred4, abs_err4 = create_rollout(model, x, x_for, x_past, s, rollout_len)

            std = (abs_err1 **2 + abs_err2**2 + abs_err3**2 + abs_err4**2)/4
            stack_pred_multiroll = torch.stack((stacked_pred1,stacked_pred2,stacked_pred3,stacked_pred4), dim=0)
            stack_pred_multiroll = torch.cat((stack_pred_multiroll, std.unsqueeze(0)), dim=0)
            stack_abserr_multiroll = torch.stack((abs_err1,abs_err2,abs_err3,abs_err4),dim=0)

            # create single rollout
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, rollout_len)

            # compute absolute difference among frames from multi rollout

            # plot multirollout Trajectories which started from same context window
            fig, axes = plt.subplots(nrows=nr_of_rollouts+1, ncols=rollout_len)
            fig.tight_layout()

            fig, (ax1, ax2, ax3) = plt.subplots(3,1)

            grid1 = torchvision.utils.make_grid(stack_pred_multiroll[0,:,...].permute(0,1,3,2).cpu(), normalize=True, nrow=1)
            ax1.imshow(grid1.permute(2,1,0)[:,:,0], cmap=color, interpolation='none')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax1.set_title('Simulated Rollout Trajectory', fontsize=15)
            cax.set_axis_off()
            ax1.axis('off')

            # grid2 = torchvision.utils.make_grid(stack_pred_multiroll[1,:,...].permute(0,1,3,2).cpu(), normalize=True, nrow=1)
            # ax2.imshow(grid2.permute(2,1,0)[:,:,0], cmap=color)
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax2.axis('off')

            # grid3 = torchvision.utils.make_grid(stack_pred_multiroll[2,:,...].permute(0,1,3,2).cpu(),normalize=True, nrow=1)
            # ax3.imshow(grid3.permute(2,1,0)[:,:,0], cmap=color)
            # divider = make_axes_locatable(ax3)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax3.axis('off')

            grid4 = torchvision.utils.make_grid(x_for.squeeze(1).permute(0,1,3,2).cpu(),normalize=True, nrow=1)
            ax2.set_title('Ground Truth', fontsize=15)
            ax2.imshow(grid4.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax2.axis('off')

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="0.8%", pad=0.05)
            grid5 = torchvision.utils.make_grid(stack_pred_multiroll[4,:,...].permute(0,1,3,2).cpu(), nrow=1)
            im5 = ax3.imshow(grid5.permute(2,1,0)[:,:,0], cmap=color)
            cbar = fig.colorbar(im5, cmap='inferno', cax=cax)
            cbar.ax.tick_params(labelsize=3)
            # cax.set_axis_off()
            ax3.set_title('Std. Dev.', fontsize=15)
            ax3.axis('off')

            plt.tight_layout()
            plt.savefig(savedir + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

            # COMPUTE METRICS
            # MAE
            mae08unorm.append(metrics.MAE(inv_scaler(stack_pred_multiroll[0,...], min_value=x_for_unorm.min(), max_value=x_for_unorm.max()), x_for_unorm).detach().cpu().numpy())
            mae08.append(metrics.MAE(stack_pred_multiroll[2,...], x_for.squeeze(1)).detach().cpu().numpy())

            # RMSE
            rmse08unorm.append(metrics.RMSE(inv_scaler(stack_pred_multiroll[0,...], min_value=x_for_unorm.min(), max_value=x_for_unorm.max()),x_for_unorm).detach().cpu().numpy())
            rmse08.append(metrics.RMSE(stack_pred_multiroll[0,...], x_for.squeeze(1)).detach().cpu().numpy())

            print(rmse08unorm[0], mae08unorm[0], rmse08[0], mae08[0])

            if batch_idx == 150:
                break

    # write results to file:
    with open(savedir_txt + 'nll_and_runtimes.txt','w') as f:
        f.write('Avrg NLL: %d \n'% np.mean(nll_list))
        f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
        f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    with open(savedir_txt + 'metric_results.txt','w') as f:

        f.write('Avrg MAE mu08:\n')
        for item in np.mean(mae08unorm, axis=0):
            f.write("%f \n" % item)

        f.write('Avrg RMSE mu08:\n')
        for item in np.mean(rmse08unorm, axis=0):
            f.write("%f \n" % item)

        # f.write("%f \n" %np.std(rmse08, axis=0))

        f.write('Norm Avrg MAE mu08:\n')
        for item in np.mean(mae08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm STD MAE mu08:\n')
        for item in np.std(mae08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm Avrg RMSE mu08:\n')
        for item in np.mean(rmse08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm STD RMSE mu08:\n')
        for item in np.std(rmse08, axis=0):
            f.write("%f \n" % item)

        # f.write("%f \n" %np.mean(mae08unorm, axis=0))
        # f.write("%f \n" %np.std(mae08unorm, axis=0))
        # f.write("%f \n" %np.std(rmse08unorm, axis=0))

    print("Average Test Neg. Log Probability Mass:", np.mean(nll_list))
    print("Average Fwd. runtime", np.mean(avrg_fwd_time))
    print("Average Bw runtime:", np.mean(avrg_bw_time))

    return np.mean(nll_list)

def test_with_ds(srmodel, stmodel, test_loader, exp_name, srmodelname, stmodelname, logstep, args):

    # Assuming 'args' is an object containing various parameters
    # such as device, s, trainset, etc.

    # Initialization of variables
    state = None
    nll_list = []
    avrg_fwd_time = []
    avrg_bw_time = []

    mae08 = []
    rmse08 = []
    rmse08unorm = []
    mae08unorm = []

    nll_st_08 = []
    nll_sr = []

    # Choose color map based on the dataset
    color = 'viridis' if args.trainset == 'geop' else 'inferno'

    # Directory for saving snapshots of the test set
    savedir = "experiments/{}_{}_{}_{}x/{}/snapshots/".format(exp_name, stmodelname, args.trainset, args.s, args.bsz)
    os.makedirs(savedir, exist_ok=True)

    # Directory for saving experiment details
    savedir_txt = 'experiments/{}_{}_{}_{}x/{}/'.format(exp_name, stmodelname, args.trainset, args.s, args.bsz)
    os.makedirs(savedir_txt, exist_ok=True)

    # Set both models to evaluation mode and disable gradient computation
    srmodel.eval()
    stmodel.eval()

    with torch.no_grad():
        # Loop through batches in the test loader
        for batch_idx, item in enumerate(test_loader):
            # Move data to the specified device
            x = item[0].to(args.device)
            x_unorm = item[1].to(args.device)

            # Split input data into past and future sequences
            x_past, x_for = x[:, :, :2, ...], x[:, :, 2:, ...]
            x_past_unorm, x_for_unorm = x_unorm[:, :2, ...], x_unorm[:, 2:, ...]

            # Resample the input for forecasting
            x_resh = F.interpolate(x[:, 0, ...], (x_for.shape[3] // args.s, x_for.shape[4] // args.s))

            # Split time series into lags and prediction window
            x_past_lr, x_for_lr = x_resh[:, :2, ...], x_resh[:, 2:, ...]

            # Reshape into the correct format [bsz, num_channels, seq_len, height, width]
            x_past_lr = x_past_lr.unsqueeze(1).contiguous().float()
            x_for_lr = x_for_lr.unsqueeze(1).contiguous().float()

            # Record the start time for measuring forecasting time
            start = timeit.default_timer()

            # Run the forecasting method
            x_for_hat_lr, state, nllst = stmodel._predict(x_past_lr.cuda(), state=None)
            x_for_hat_lr = x_for_hat_lr.squeeze(1)

            # Super-resolve the result
            x_for_hat, nllsr = srmodel(xlr=x_for_hat_lr, reverse=True, eps=0.8)

            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)
            avrg_fwd_time.append(stop-start)

            # ---------------------- Evaluate Predictions---------------------- #

            # Evalutae for different temperatures
            # mu0 = model._predict(x_past, state, eps=0)
            # mu05 = model._predict(x_past, state, eps=0.5)
            # mu08 = model._predict(x_past, state, eps=0.8)
            # mu1 = model._predict(x_past, state, eps=1)

            print(" Create Rollouts ... ")
            rollout_len = args.bsz - 1
            eps = 0.8

            start = timeit.default_timer() # TODO put timing function somewhere else

            stop = timeit.default_timer()

            print("Time Bwd pass / predicting:", stop - start)
            avrg_bw_time.append(stop - start)

            # create multiple rollouts with same initial conditions
            nr_of_rollouts = 4
            stacked_pred1, abs_err1, nll_st_1 = create_rollout(stmodel, x_for_lr, x_past_lr, rollout_len)
            stacked_pred2, abs_err2, nll_st_2 = create_rollout(stmodel, x_for_lr, x_past_lr, rollout_len)
            stacked_pred3, abs_err3, nll_st_3 = create_rollout(stmodel, x_for_lr, x_past_lr, rollout_len)
            stacked_pred4, abs_err4, nll_st_4 = create_rollout(stmodel, x_for_lr, x_past_lr, rollout_len)

            # super-resolve predictions
            stacked_pred1, _ = srmodel(xlr=stacked_pred1, eps=eps, reverse=True)
            stacked_pred2, _ = srmodel(xlr=stacked_pred2, eps=eps, reverse=True)
            stacked_pred3, _ = srmodel(xlr=stacked_pred3, eps=eps, reverse=True)
            stacked_pred4, _ = srmodel(xlr=stacked_pred4, eps=eps, reverse=True)

            # compute absolute error of super-resolved predictions
            abs_err1 = torch.abs(stacked_pred1.cuda() - x_for[:,...].cuda())
            abs_err2 = torch.abs(stacked_pred2.cuda() - x_for[:,...].cuda())
            abs_err3 = torch.abs(stacked_pred3.cuda() - x_for[:,...].cuda())
            abs_err4 = torch.abs(stacked_pred4.cuda() - x_for[:,...].cuda())

            std = (abs_err1 **2 + abs_err2**2 + abs_err3**2 + abs_err4**2)/4

            stack_pred_multiroll = torch.stack((stacked_pred1,stacked_pred2,stacked_pred3,stacked_pred4), dim=0)
            stack_pred_multiroll = torch.cat((stack_pred_multiroll, std), dim=0)
            stack_abserr_multiroll = torch.stack((abs_err1,abs_err2,abs_err3,abs_err4),dim=0)

            # Plot Multirollout trajectories which started from same context window
            fig, axes = plt.subplots(nrows=nr_of_rollouts+1, ncols=rollout_len, constrained_layout = True)

            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            grid1 = torchvision.utils.make_grid(stack_pred_multiroll[0,:,...].permute(0,1,3,2).cpu(), normalize=True, nrow=1)
            ax1.imshow(grid1.permute(2,1,0)[:,:,0], cmap=color, interpolation='none')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax1.set_title('Simulated Rollout Trajectory', fontsize=15)
            cax.set_axis_off()
            ax1.axis('off')

            # grid2 = torchvision.utils.make_grid(stack_pred_multiroll[1,:,...].permute(0,1,3,2).cpu(), normalize=True, nrow=1)
            # ax2.imshow(grid2.permute(2,1,0)[:,:,0], cmap=color)
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax2.axis('off')

            # grid3 = torchvision.utils.make_grid(stack_pred_multiroll[2,:,...].permute(0,1,3,2).cpu(),normalize=True, nrow=1)
            # ax3.imshow(grid3.permute(2,1,0)[:,:,0], cmap=color)
            # divider = make_axes_locatable(ax3)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax3.axis('off')

            grid4 = torchvision.utils.make_grid(x_for.squeeze(1).permute(0,1,3,2).cpu(),normalize=True, nrow=1)
            ax2.set_title('Ground Truth', fontsize=15)
            ax2.imshow(grid4.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax2.axis('off')

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            grid5 = torchvision.utils.make_grid(stack_pred_multiroll[4,:,...].permute(0,1,3,2).cpu(), nrow=1)
            im5 = ax3.imshow(grid5.permute(2,1,0)[:,:,0], cmap=color)
            cbar = fig.colorbar(im5, cmap='inferno', cax=cax)
            cbar.ax.tick_params(labelsize=3)
            # cax.set_axis_off()
            ax3.set_title('Std. Dev.', fontsize=15)
            ax3.axis('off')

            plt.show()
            plt.tight_layout()
            plt.savefig(savedir + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

            # COMPUTE METRICS
            # MAE
            mae08unorm.append(metrics.MAE(inv_scaler(stack_pred_multiroll[0,...], min_value=x_for_unorm.min(), max_value=x_for_unorm.max()), x_for_unorm).detach().cpu().numpy())
            mae08.append(metrics.MAE(stack_pred_multiroll[2,...], x_for.squeeze(1)).detach().cpu().numpy())

            # RMSE
            rmse08unorm.append(metrics.RMSE(inv_scaler(stack_pred_multiroll[0,...], min_value=x_for_unorm.min(), max_value=x_for_unorm.max()),x_for_unorm).detach().cpu().numpy())
            rmse08.append(metrics.RMSE(stack_pred_multiroll[0,...], x_for.squeeze(1)).detach().cpu().numpy())

            # NLL
            nll_st_08.append(nll_st_1)

            print('Unorm RMSE, MAE score', rmse08unorm[0].mean(), mae08unorm[0].mean())
            print('Norm RMSE, MAE score:', rmse08[0].mean(), mae08[0].mean())

            if batch_idx == 150:
                break

            # TODO add CRPS score

    # write results to file:
    with open(savedir_txt + 'nll_and_runtimes.txt','w') as f:
        # f.write('Avrg NLL: %d \n'% np.mean(nll_list))
        f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
        f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    with open(savedir_txt + 'metric_results.txt','w') as f:

        f.write('Avrg MAE mu08:\n')
        for item in np.mean(mae08unorm, axis=0):
            f.write("%f \n" % item)

        f.write('Avrg RMSE mu08:\n')
        for item in np.mean(rmse08unorm, axis=0):
            f.write("%f \n" % item)

        # f.write("%f \n" %np.std(rmse08, axis=0))

        f.write('Norm Avrg MAE mu08:\n')
        for item in np.mean(mae08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm STD MAE mu08:\n')
        for item in np.std(mae08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm Avrg RMSE mu08:\n')
        for item in np.mean(rmse08, axis=0):
            f.write("%f \n" % item)

        f.write('Norm STD RMSE mu08:\n')
        for item in np.std(rmse08, axis=0):
            f.write("%f \n" % item)

        f.write('STD NLL:\n')
        for item in np.std(nll_st_08, axis=0):
            f.write("%f \n" % item)

        f.write('MEAN NLL:\n')
        for item in np.mean(nll_st_08, axis=0):
            f.write("%f \n" % item)

    return None #np.mean(nll_list)

def metrics_eval(args, model, test_loader, exp_name, modelname, logstep):
    """
    Note: batch size indicates lead time we predict.
    """

    print("Metric evaluation on {}...".format(args.trainset))

    # storing metrics
    mae08 = []
    mse08 = []
    rmse08 = []
    nll08 = []

    state = None

    # creat and save metric plots
    savedir = "experiments/{}/plots/test_set_{}/".format(exp_name, args.trainset)
    os.makedirs(savedir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]
            x_unorm = item[1]

            # split time series into lags and prediction window
            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]
            x_past_unorm, x_for_unorm = x_unorm[:,:-1,...].float().cuda(), x_unorm[:,-1,:,:,:].unsqueeze(1).cuda().float()

            x_past = x_past.permute(0,2,1,3,4).contiguous().float().to(args.device)
            x_for = x_for.permute(0,2,1,3,4).contiguous().float().to(args.device)
            x_for_unorm = x_for_unorm.permute(0,2,1,3,4).contiguous().float().to(args.device)

            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)

            # track metric over forecasting period
            print("Forecast ... ")
            lead_time = args.bsz-1
            eps = 0.8
            predictions = []
            past = x_past[0,:,:,:,:].unsqueeze(0)

            x, s, nll = model._predict(x_past=past, # TODO return nll
                                  state=None,
                                  eps=eps)

            print('COMPUTING ROLLOUT!')
            rollout_len = 0
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, lead_time)
            x = stacked_pred

            print('ROLLOUT COMPUTED!')

            # MAE
            mae08.append(metrics.MAE(inv_scaler(stacked_pred, min_value=x_for_unorm.min(), max_value=x_for_unorm.max()), x_for_unorm).detach().cpu().numpy())

            # RMSE
            rmse08.append(metrics.RMSE(inv_scaler(stacked_pred, min_value=x_for_unorm.min(), max_value=x_for_unorm.max()),x_for_unorm).detach().cpu().numpy())

            # NLL
            nll08.append(nll)

            print('3 h', current_rmse[3], current_psnr[3], current_ssim[3])
            print('20 h', current_rmse[20], current_psnr[20], current_ssim[20])

            if batch_idx == 200:
                print(batch_idx)
                break


        # compute average SSIM for each temperature map on predicted day t
        avrg_ssim = list(map(lambda x: x/200, ssim))#len(test_loader), ssim))

        # compute average PSNR for each temperature map on predicted day t
        avrg_psnr = list(map(lambda x: x/200, psnr))#len(test_loader), psnr))

        avrg_mmd = list(map(lambda x: x/200, mmd))#len(test_loader), mmd))

        avrg_emd = list(map(lambda x: x/200, emd))#len(test_loader), emd))

        avrg_rmse = list(map(lambda x: x/200, rmse))#len(test_loader), rmse))

        plt.plot(avrg_ssim, label='ST-Flow Best SSIM', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average SSIM')
        plt.savefig(savedir + '/avrg_ssim.png', dpi=300)
        plt.close()

        plt.plot(avrg_psnr, label='ST-Flow Best PSNR', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average PSNR')
        plt.savefig(savedir + '/avrg_psnr.png', dpi=300)
        plt.close()

        plt.plot(avrg_mmd, label='ST-Flow Best MMD', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average MMD')
        plt.savefig(savedir + '/avrg_mmd.png', dpi=300)
        plt.close()

        plt.plot(avrg_emd, label='ST-Flow Best EMD', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average EMD')
        plt.savefig(savedir + '/avrg_emd.png', dpi=300)
        plt.close()

        plt.plot(avrg_rmse, label='ST-Flow Best RMSE', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average RMSE')
        plt.savefig(savedir + '/avrg_rmse.png', dpi=300)
        plt.close()

        # Write metric results to a file in case to recreate plots
        with open(savedir + 'metric_results.txt','w') as f:
            f.write('Avrg SSIM over forecasting period:\n')
            for item in avrg_ssim:
                f.write("%f \n" % item)

            f.write('Avrg PSNR over forecasting period:\n')
            for item in avrg_psnr:
                f.write("%f \n" % item)

            f.write('Avrg MMD over forecasting period:\n')
            for item in avrg_mmd:
                f.write("%f \n" % item)

            f.write('Avrg EMD over forecasting period:\n')
            for item in avrg_emd:
                f.write("%f \n" % item)

            f.write('Avrg RMSE over forecasting period:\n')
            for item in avrg_rmse:
                f.write("%f \n" % item)

        return None

def metrics_eval_all(roll_len=30):

    print("Creating unified plot from text files ...")

    path = os.getcwd()

    def read_metrics(fname):

        avrg_rmse = []
        std_rmse = []
        avrg_mae = []
        std_mae = []
        lines = []

        # read metric results from file
        print('Reading file:', fname)
        with open(path + fname, 'r') as f:
            line = f.readline()

            while line != '':
                print(line, end='')
                line = f.readline()

                if line == 'Norm Avrg RMSE mu08:\n':
                    avrg_mae = lines
                    lines = []
                    continue

                if line == 'Norm STD MAE mu08:\n':
                    avrg_rmse = lines
                    lines = []
                    continue

                if line == 'Norm STD RMSE mu08:\n':
                    std_mae = lines
                    lines = []
                    continue

                if line == '':
                    pass

                else:
                    lines.append(float(line))

            std_rmse = lines
        return avrg_rmse, std_rmse, avrg_mae, std_mae

    # avrg_rmse_nods, std_rmse_nods, _, _ = read_metrics('/flow-3-level-2-k_model_epoch_1_step_34250_wbench_nods/100STD/metric_results_normalized.txt')
    if args.trainset == 'geop':
        avrg_rmse_16x, std_rmse_16x,_, _ = read_metrics('/experiments/flow-1-level-2-k_model_epoch_2_step_43750_geop_16x/{}/metric_results_normalized.txt'.format(roll_len))
        avrg_rmse_8x, std_rmse_8x,_, _ = read_metrics('/experiments/flow-2-level-2-k_model_epoch_2_step_41000_geop_8x/{}/metric_results_normalized.txt'.format(roll_len))
        avrg_rmse_4x, std_rmse_4x,_, _ = read_metrics('/experiments/flow-3-level-2-k_model_epoch_1_step_22750_geop_4x/{}/metric_results_normalized.txt'.format(roll_len))

    elif args.trainset == 'temp':
        avrg_rmse_16x, std_rmse_16x,_, _ = read_metrics('/experiments/flow-3-level-2-k_model_epoch_5_step_3750_temp_16x/{}/metric_results_normalized.txt'.format(roll_len))
        avrg_rmse_8x, std_rmse_8x,_, _ = read_metrics('/experiments/flow-3-level-2-k_model_epoch_8_step_5500_temp_8x/{}/metric_results_normalized.txt'.format(roll_len))
        avrg_rmse_4x, std_rmse_4x,_, _ = read_metrics('/experiments/flow-3-level-2-k_model_epoch_7_step_4750_temp_4x/{}/metric_results_normalized.txt'.format(roll_len))


    # pdb.set_trace()
    avrg_rmse_16x = np.array(avrg_rmse_16x)
    error16x = np.array(std_rmse_16x)

    avrg_rmse_8x = np.array(avrg_rmse_8x)
    error8x = np.array(std_rmse_16x)

    avrg_rmse_4x = np.array(avrg_rmse_4x)
    error4x = np.array(std_rmse_4x)

    # error = np.array(std_rmse)

    xticks = np.arange(0,roll_len,1)

    plt.plot(avrg_rmse_16x, label='ST-Flow - 16x', color='darkviolet')
    plt.fill_between(xticks, avrg_rmse_16x - error16x, avrg_rmse_16x + error16x, color='gray', alpha=0.2)

    plt.plot(avrg_rmse_8x, label='ST-Flow - 8x', color='deeppink')
    plt.fill_between(xticks, avrg_rmse_8x - error8x, avrg_rmse_8x + error8x, color='gray', alpha=0.2)

    plt.plot(avrg_rmse_4x, label='ST-Flow - 4x', color='mediumslateblue')
    plt.fill_between(xticks, avrg_rmse_4x - error4x, avrg_rmse_4x + error4x, color='gray', alpha=0.2)

    # plt.plot(avrg_rmse_l3k3, label='ST-Flow L-3 K-3', color='mediumslateblue')
    # plt.plot(avrg_rmse_3dunet, label='3DUnet', color='lightseagreen')
    plt.grid(axis='y')
    plt.axvline(x=2, color='orangered')
    plt.legend(loc='best')
    plt.xlabel('Time-Step')
    plt.ylabel('Average RMSE')
    plt.title('T2M')
    plt.show()
    plt.savefig(path + '/avrg_rmse_all_temp_norm.png') #, dpi=300)
    plt.close()


if __name__ == "__main__":

    print(torch.cuda.device_count())

    # NOTE: when executing code, make sure you enable the --testmode flag !!!

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    height, width = next(iter(test_loader))[0].shape[3], next(iter(test_loader))[0].shape[4]

    args.device = "cuda"

    metrics_eval_all()

    if args.ds or args.s > 1: # simulation run on downsampled / embedded representation

        # load model
        if args.trainset == 'geop':

            if args.s == 4:
                srmodelname = 'model_epoch_1_step_21250'
                srmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_wbench_2023_12_06_07_29_49/srmodel_checkpoints/{}.tar'.format(srmodelname)
                stmodelname = 'model_epoch_1_step_21250'
                stmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_wbench_2023_12_06_07_29_49/stmodel_checkpoints/{}.tar'.format(stmodelname)
                
            elif args.s == 8:
                srmodelname = 'model_epoch_0_step_6000'
                srmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_geop_8x_2024_01_26_12_21_42/srmodel_checkpoints/{}.tar'.format(srmodelname)
                stmodelname = 'model_epoch_0_step_6000'
                stmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_geop_8x_2024_01_26_12_21_42/srmodel_checkpoints/{}.tar'.format(stmodelname)
        
            # elif args.s == 16:
            #    srmodelname = 
            #    srmodelpath = 

        if args.trainset == 'temp':

            if args.s == 4:
                srmodelname = 'model_epoch_7_step_4750'
                srmodelpath = '/home/mila/c/christina.winkler/scratch/climsim_exp_jan2024/flow_temp_4x_2024_01_26_12_21_40/srmodel_checkpoints/{}.tar'.format(srmodelname)
                stmodelname = 'model_epoch_7_step_4750'
                stmodelpath = '/home/mila/c/christina.winkler/scratch/climsim_exp_jan2024/flow_temp_4x_2024_01_26_12_21_40/stmodel_checkpoints/{}.tar'.format(stmodelname)

            elif args.s == 8:
                srmodelname = 'model_epoch_0_step_6000'
                srmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_geop_8x_2024_01_26_12_21_42/srmodel_checkpoints/{}.tar'.format(srmodelname)
                stmodelname = 'model_epoch_0_step_6000'
                stmodelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_geop_8x_2024_01_26_12_21_42/srmodel_checkpoints/{}.tar'.format(stmodelname)
        
            # elif args.s == 16:
            #    srmodelname = 
            #    srmodelpath = 
        
        srmodel = srflow.SRFlow((in_channels, height, width), args.filter_size, 3, 2,
                                    args.bsz, args.s, args.nb, args.condch, args.nbits,
                                    args.noscale, args.noscaletest).to(args.device)

        srckpt = torch.load(srmodelpath, map_location='cuda:0')
        srmodel.load_state_dict(srckpt['model_state_dict'])
        srmodel.eval()

        stmodel = condNF.FlowModel((in_channels, height//args.s, width//args.s),
                                args.filter_size, args.Lst, args.Kst, args.bsz,
                                args.lag_len, args.s, args.nb, args.device,
                                args.condch, args.nbits,
                                args.noscale, args.noscaletest, args.testmode).to(args.device)

        stckpt = torch.load(stmodelpath, map_location='cuda:0')
        stmodel.load_state_dict(stckpt['model_state_dict'])
        stmodel.eval()

        srparams = sum(x.numel() for x in srmodel.parameters() if x.requires_grad)
        stparams = sum(x.numel() for x in stmodel.parameters() if x.requires_grad)
        params = srparams + stparams
        print('Nr of Trainable Params SR {}:  '.format(args.device), srparams)
        print('Nr of Trainable Params ST {}:  '.format(args.device), stparams)
        print('Total Nr of Trainable Params ST {}:  '.format(args.device), params)

        print("Evaluate on test split with DS ...")
        test_with_ds(srmodel, stmodel, test_loader, "flow-{}-level-{}-k".format(args.Lst, args.Kst), srmodelname, stmodelname, -999999999, args)
        # metrics_eval(args, model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999)

    else:
        # no downscaling, simulation run on original input size
        modelname = 'model_epoch_1_step_34250'
        modelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_wbench_no_ds__2023_12_05_05_51_46/model_checkpoints/{}.tar'.format(modelname)

        model = condNF.FlowModel((in_channels, height, width),
                                args.filter_size, args.Lst, args.Kst, args.bsz,
                                args.lag_len, args.s, args.nb, args.device,
                                args.condch, args.nbits,
                                args.noscale, args.noscaletest, args.testmode)

        ckpt = torch.load(modelpath, map_location='cuda:0')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        params = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print('Nr of Trainable Params {}:  '.format(args.device), params)
        print("Evaluate on test split ...")
        # test(model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.Lst, args.Kst), modelname, -99999, args)
        # metrics_eval(args, model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999)
        # metrics_eval_all()
