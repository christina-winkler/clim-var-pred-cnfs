import numpy as np
import torch
import random

import PIL
import os
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

from models.architectures import condNF
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
                        help="Interval in which results should be logged.")
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
parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
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
parser.add_argument("--L", type=int, default=3, help="# of levels")
parser.add_argument("--K", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128//8,
                        help="# of residual-in-residual blocks in LR network.")

# data
parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
                        help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="wbench",
                        help="Dataset to train the model on.")

args = parser.parse_args()

def inv_scaler(x, min_value=0, max_value=100):
    x = x * (max_value - min_value) + min_value
    return x

def create_rollout(model, init_pred, x_for, x_past, s, lead_time):

    predictions = []
    predictions.append(init_pred[0,:,:,:,:])
    interm = x_past[0,:,1,:,:].unsqueeze(1).cuda()

    for l in range(lead_time):
        context = torch.cat((predictions[l-1], interm), 1)
        x, s = model._predict(x_past=context.unsqueeze(0), state=s)
        predictions.append(x[0,:,:,:,:])
        interm = x[0,:,:,:,:] # update intermediate state

    stacked_pred = torch.stack(predictions, dim=0).squeeze(1).squeeze(2)

    # compute absolute error images
    abs_err = torch.abs(stacked_pred.cuda() - x_for[:,...].cuda().squeeze(1))

    return stacked_pred, abs_err

def test(model, test_loader, exp_name, modelname, logstep, args):

    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    state=None
    nll_list=[]
    avrg_fwd_time = []
    avrg_bw_time = []
    model.eval()
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    savedir = "{}_{}/snapshots/test_set_{}/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]
            time, lat, lon = item[1], item[2], item[3]

            # split time series into lags and prediction window
            x_past, x_for = x[:,:-1,...], x[:,-1,:,:,:].unsqueeze(1)

            x_past = x_past.permute(0,2,1,3,4).contiguous().float()
            x_for = x_for.permute(0,2,1,3,4).contiguous().float()

            start = timeit.default_timer()
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

            # # create multiple rollouts with same initial conditions
            # pred_multiroll = []
            # abs_err_multiroll = []
            nr_of_rollouts = 4
            # for i in range(nr_of_rollouts):
            #     pred, err = create_rollout(model, x, x_for, x_past, s, rollout_len)
            #     pred_multiroll.append(pred.squeeze(1))
            #     abs_err_multiroll.append(err.squeeze(1))

            stacked_pred1, abs_err1 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred2, abs_err2 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred3, abs_err3 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred4, abs_err4 = create_rollout(model, x, x_for, x_past, s, rollout_len)

            std = (abs_err1 **2 + abs_err2**2 + abs_err3**2 + abs_err4**2)/4

            # stack_pred_multiroll = torch.stack(pred_multiroll, dim=0)
            stack_pred_multiroll = torch.stack((stacked_pred1,stacked_pred2,stacked_pred3,stacked_pred4), dim=0)
            stack_pred_multiroll = torch.cat((stack_pred_multiroll, std.unsqueeze(0)), dim=0)
            # stack_abserr_multiroll = torch.stack(abs_err_multiroll, dim=0)
            stack_abserr_multiroll = torch.stack((abs_err1,abs_err2,abs_err3,abs_err4),dim=0)

            # create single rollout
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, rollout_len)

            # compute absolute difference among frames from multi rollout

            # Plot Multirollout Trajectories which started from same context window
            fig, axes = plt.subplots(nrows=nr_of_rollouts+1, ncols=rollout_len)

            # single_pred = torchvision.utils.make_grid(stacked_pred[h-1,:,:,:].squeeze(1).cpu(), nrow=1)
            # single_pred = single_pred[0,:,:].transpose(0,1)
            # plt.imshow(single_pred, cmap=color, extent=[0,350,-80,85],
                       # interpolation='none')

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)
            grid1 = torchvision.utils.make_grid(stack_pred_multiroll[0,:,...].cpu(), normalize=True, nrow=1)
            # norm = ImageNormalize(grid1, interval=MinMaxInterval(), stretch=SqrtStretch())
            ax1.imshow(grid1.permute(2,1,0)[:,:,0], cmap=color, interpolation='none')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax1.set_title('Simulated Rollout Trajectories', fontsize=5)
            cax.set_axis_off()
            ax1.axis('off')

            grid2 = torchvision.utils.make_grid(stack_pred_multiroll[1,:,...].cpu(), normalize=True, nrow=1)
            ax2.imshow(grid2.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax2.axis('off')

            grid3 = torchvision.utils.make_grid(stack_pred_multiroll[2,:,...].cpu(),normalize=True, nrow=1)
            ax3.imshow(grid3.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax3.axis('off')

            grid4 = torchvision.utils.make_grid(x_for.squeeze(1).cpu(),normalize=True, nrow=1)
            ax4.set_title('Ground Truth', fontsize=5)
            ax4.imshow(grid4.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax4.axis('off')

            divider = make_axes_locatable(ax5)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            grid5 = torchvision.utils.make_grid(stack_pred_multiroll[4,:,...].cpu(), nrow=1)
            im5 = ax5.imshow(grid5.permute(2,1,0)[:,:,0], cmap=color)
            cbar = fig.colorbar(im5, cmap='inferno', cax=cax)
            cbar.ax.tick_params(labelsize=3)
            # cax.set_axis_off()
            ax5.set_title('Std. Dev.', fontsize=5)
            ax5.axis('off')

            plt.tight_layout()
            plt.savefig(savedir + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()


            # Plot differences of rollout trajectories
            # compare different frames from same rollout, should give a difference picture
            test_diff = stack_pred_multiroll.squeeze(1)[0,1,0,...] - stack_pred_multiroll.squeeze(1)[0,2,0,...]
            plt.figure()
            plt.imshow(test_diff.unsqueeze(2).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 0")
            plt.savefig(savedir + "tesdiff1_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            # compare same prediction time step from different rollouts
            test_diff = stack_pred_multiroll.squeeze(1)[0,1,0,...] - stack_pred_multiroll.squeeze(1)[2,1,0,...]
            plt.figure()
            plt.imshow(test_diff.unsqueeze(2).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 1")
            # plt.show()
            plt.savefig(savedir + "tesdiff2_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            # pdb.set_trace()
            grid_ground_truth = torchvision.utils.make_grid(x_for[0:6, :, :, :].squeeze(1).cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_ground_truth.permute(2, 1, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            plt.title("Ground Truth at t (test)")
            plt.savefig(savedir + "gt_x_t+1_step_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            # visualize past frames the prediction is based on (context)
            grid_past = torchvision.utils.make_grid(x_past[0:6, -1, :, :].cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_past.permute(2, 1, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            plt.title("Context window (test)")
            plt.savefig(savedir + "context_step_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            grid_predictions = torchvision.utils.make_grid(stacked_pred[0:6,:,:,:].cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_predictions.permute(2, 1, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu={}".format(eps))
            plt.tight_layout()
            plt.savefig(savedir + "prediction_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            grid_abs_error = torchvision.utils.make_grid(abs_err[0:6,:,:,:].cpu(), nrow=1)
            plt.figure()
            plt.imshow(grid_abs_error.permute(2, 1, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Absolute Error (test)")
            plt.tight_layout()
            plt.savefig(savedir + "absolute_error_logstep_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            # visualize single prediction
            plt.figure()
            h = 1
            single_pred = torchvision.utils.make_grid(stacked_pred[h-1,:,:,:].squeeze(1).cpu(), normalize=True, nrow=1)
            single_pred = single_pred[0,:,:].transpose(0,1)

            # https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
            # https://stackoverflow.com/questions/18696122/change-values-on-matplotlib-imshow-graph-axis
            # rotated_pred = ndimage.rotate(single_pred, 90)
            # pdb.set_trace()
            plt.imshow(single_pred, cmap=color, extent=[0,350,-80,85],
                       interpolation='none')
            plt.colorbar(label=r'Geopotential [$m^2 s^{-2}$]', shrink=0.6)
            plt.xlabel('longitude')
            # plt.axes(projection=ccrs.Orthographic(central_longitude=20, central_latitude=40))
            # plt.xlim([0, 350])
            # plt.xticks(np.arange(0,350,50))
            plt.ylabel('latitude')
            # plt.ylim([-80, 80])
            # plt.axis('off')
            plt.title(r"Prediction at t={}h ahead, mu={}".format(h, eps))
            plt.savefig(savedir + "single_prediction_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)

            # plt.show()

            # grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
            # plt.figure()
            # plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap='virdis')
            # plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.8")
            # plt.savefig(savedir + "mu_0.8_logstep_{}_test.png".format(logstep), dpi=300)
            #
            # grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
            # plt.figure()
            # plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
            # plt.axis('off')
            # plt.title("Prediction at t (test), mu=1.0")
            # plt.savefig(savedir + "mu_1_logstep_{}_test.png".format(logstep), dpi=300)
            plt.close()

            # write results to file:
            with open('{}_{}/nll_runtimes.txt'.format(exp_name, modelname),'w') as f:
                f.write('Avrg NLL: %d \n'% np.mean(nll_list))
                f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
                f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    print("Average Test Neg. Log Probability Mass:", np.mean(nll_list))
    print("Average Fwd. runtime", np.mean(avrg_fwd_time))
    print("Average Bw runtime:", np.mean(avrg_bw_time))

    return np.mean(nll_list)

def test_with_ds(srmodel, stmodel, test_loader, exp_name, modelname, logstep, args):

    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    state=None
    nll_list=[]
    avrg_fwd_time = []
    avrg_bw_time = []
    model.eval()
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    savedir = "{}_{}/snapshots/test_set_{}/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]
            time, lat, lon = item[1], item[2], item[3]

            # split time series into lags and prediction window
            x_past, x_for = x[:,:-1,...], x[:,-1,:,:,:].unsqueeze(1)

            x_past = x_past.permute(0,2,1,3,4).contiguous().float()
            x_for = x_for.permute(0,2,1,3,4).contiguous().float()

            start = timeit.default_timer()
            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)
            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)
            avrg_fwd_time.append(stop-start)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            # ---------------------- Evaluate Predictions---------------------- #

            # Evalutae for different temperatures
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

            # # create multiple rollouts with same initial conditions
            # pred_multiroll = []
            # abs_err_multiroll = []
            nr_of_rollouts = 4
            # for i in range(nr_of_rollouts):
            #     pred, err = create_rollout(model, x, x_for, x_past, s, rollout_len)
            #     pred_multiroll.append(pred.squeeze(1))
            #     abs_err_multiroll.append(err.squeeze(1))

            stacked_pred1, abs_err1 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred2, abs_err2 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred3, abs_err3 = create_rollout(model, x, x_for, x_past, s, rollout_len)
            stacked_pred4, abs_err4 = create_rollout(model, x, x_for, x_past, s, rollout_len)

            std = (abs_err1 **2 + abs_err2**2 + abs_err3**2 + abs_err4**2)/4

            # stack_pred_multiroll = torch.stack(pred_multiroll, dim=0)
            stack_pred_multiroll = torch.stack((stacked_pred1,stacked_pred2,stacked_pred3,stacked_pred4), dim=0)
            stack_pred_multiroll = torch.cat((stack_pred_multiroll, std.unsqueeze(0)), dim=0)
            # stack_abserr_multiroll = torch.stack(abs_err_multiroll, dim=0)
            stack_abserr_multiroll = torch.stack((abs_err1,abs_err2,abs_err3,abs_err4),dim=0)

            # create single rollout
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, rollout_len)

            # compute absolute difference among frames from multi rollout

            # Plot Multirollout Trajectories which started from same context window
            fig, axes = plt.subplots(nrows=nr_of_rollouts+1, ncols=rollout_len)

            # single_pred = torchvision.utils.make_grid(stacked_pred[h-1,:,:,:].squeeze(1).cpu(), nrow=1)
            # single_pred = single_pred[0,:,:].transpose(0,1)
            # plt.imshow(single_pred, cmap=color, extent=[0,350,-80,85],
                       # interpolation='none')

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)
            grid1 = torchvision.utils.make_grid(stack_pred_multiroll[0,:,...].cpu(), normalize=True, nrow=1)
            # norm = ImageNormalize(grid1, interval=MinMaxInterval(), stretch=SqrtStretch())
            ax1.imshow(grid1.permute(2,1,0)[:,:,0], cmap=color, interpolation='none')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax1.set_title('Simulated Rollout Trajectories', fontsize=5)
            cax.set_axis_off()
            ax1.axis('off')

            grid2 = torchvision.utils.make_grid(stack_pred_multiroll[1,:,...].cpu(), normalize=True, nrow=1)
            ax2.imshow(grid2.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax2.axis('off')

            grid3 = torchvision.utils.make_grid(stack_pred_multiroll[2,:,...].cpu(),normalize=True, nrow=1)
            ax3.imshow(grid3.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax3.axis('off')

            grid4 = torchvision.utils.make_grid(x_for.squeeze(1).cpu(),normalize=True, nrow=1)
            ax4.set_title('Ground Truth', fontsize=5)
            ax4.imshow(grid4.permute(2,1,0)[:,:,0], cmap=color)
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax4.axis('off')

            divider = make_axes_locatable(ax5)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            grid5 = torchvision.utils.make_grid(stack_pred_multiroll[4,:,...].cpu(), nrow=1)
            im5 = ax5.imshow(grid5.permute(2,1,0)[:,:,0], cmap=color)
            cbar = fig.colorbar(im5, cmap='inferno', cax=cax)
            cbar.ax.tick_params(labelsize=3)
            # cax.set_axis_off()
            ax5.set_title('Std. Dev.', fontsize=5)
            ax5.axis('off')

            plt.tight_layout()
            plt.savefig(savedir + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()


            # Plot differences of rollout trajectories
            # compare different frames from same rollout, should give a difference picture
            test_diff = stack_pred_multiroll.squeeze(1)[0,1,0,...] - stack_pred_multiroll.squeeze(1)[0,2,0,...]
            plt.figure()
            plt.imshow(test_diff.unsqueeze(2).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 0")
            plt.savefig(savedir + "tesdiff1_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            # compare same prediction time step from different rollouts
            test_diff = stack_pred_multiroll.squeeze(1)[0,1,0,...] - stack_pred_multiroll.squeeze(1)[2,1,0,...]
            plt.figure()
            plt.imshow(test_diff.unsqueeze(2).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 1")
            # plt.show()
            plt.savefig(savedir + "tesdiff2_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            # pdb.set_trace()
            grid_ground_truth = torchvision.utils.make_grid(x_for[0:6, :, :, :].squeeze(1).cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_ground_truth.permute(2, 1, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            plt.title("Ground Truth at t (test)")
            plt.savefig(savedir + "gt_x_t+1_step_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            # visualize past frames the prediction is based on (context)
            grid_past = torchvision.utils.make_grid(x_past[0:6, -1, :, :].cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_past.permute(2, 1, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            plt.title("Context window (test)")
            plt.savefig(savedir + "context_step_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            grid_predictions = torchvision.utils.make_grid(stacked_pred[0:6,:,:,:].cpu(), normalize=True, nrow=1)
            plt.figure()
            plt.imshow(grid_predictions.permute(2, 1, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu={}".format(eps))
            plt.tight_layout()
            plt.savefig(savedir + "prediction_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)
            plt.close()

            grid_abs_error = torchvision.utils.make_grid(abs_err[0:6,:,:,:].cpu(), nrow=1)
            plt.figure()
            plt.imshow(grid_abs_error.permute(2, 1, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Absolute Error (test)")
            plt.tight_layout()
            plt.savefig(savedir + "absolute_error_logstep_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            # visualize single prediction
            plt.figure()
            h = 1
            single_pred = torchvision.utils.make_grid(stacked_pred[h-1,:,:,:].squeeze(1).cpu(), normalize=True, nrow=1)
            single_pred = single_pred[0,:,:].transpose(0,1)

            # https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
            # https://stackoverflow.com/questions/18696122/change-values-on-matplotlib-imshow-graph-axis
            # rotated_pred = ndimage.rotate(single_pred, 90)
            # pdb.set_trace()
            plt.imshow(single_pred, cmap=color, extent=[0,350,-80,85],
                       interpolation='none')
            plt.colorbar(label=r'Geopotential [$m^2 s^{-2}$]', shrink=0.6)
            plt.xlabel('longitude')
            # plt.axes(projection=ccrs.Orthographic(central_longitude=20, central_latitude=40))
            # plt.xlim([0, 350])
            # plt.xticks(np.arange(0,350,50))
            plt.ylabel('latitude')
            # plt.ylim([-80, 80])
            # plt.axis('off')
            plt.title(r"Prediction at t={}h ahead, mu={}".format(h, eps))
            plt.savefig(savedir + "single_prediction_mu_{}_logstep_{}_test.png".format(eps, batch_idx), dpi=300)

            # plt.show()

            # grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
            # plt.figure()
            # plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap='virdis')
            # plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.8")
            # plt.savefig(savedir + "mu_0.8_logstep_{}_test.png".format(logstep), dpi=300)
            #
            # grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
            # plt.figure()
            # plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
            # plt.axis('off')
            # plt.title("Prediction at t (test), mu=1.0")
            # plt.savefig(savedir + "mu_1_logstep_{}_test.png".format(logstep), dpi=300)
            plt.close()

            # write results to file:
            with open('{}_{}/nll_runtimes.txt'.format(exp_name, modelname),'w') as f:
                f.write('Avrg NLL: %d \n'% np.mean(nll_list))
                f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
                f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    print("Average Test Neg. Log Probability Mass:", np.mean(nll_list))
    print("Average Fwd. runtime", np.mean(avrg_fwd_time))
    print("Average Bw runtime:", np.mean(avrg_bw_time))

    return np.mean(nll_list)

def metrics_eval(args, model, test_loader, exp_name, modelname, logstep):
    """
    Note: batch size indicates lead time we predict.
    """

    print("Metric evaluation on {}...".format(args.trainset))

    # storing metrics
    ssim = [0] * args.bsz
    psnr = [0] * args.bsz
    mmd = [0] * args.bsz
    emd = [0] * args.bsz
    rmse = [0] * args.bsz

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
            x_past, x_for = x[:,:-1,...], x[:,-1,:,:,:].unsqueeze(1)
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

            x, s = model._predict(x_past=past,
                                  state=None,
                                  eps=eps)

            print('COMPUTING ROLLOUT!')
            rollout_len = 0
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, lead_time)
            x = stacked_pred

            print('ROLLOUT COMPUTED!')

            # # SSIM
            current_ssim = metrics.ssim(x, x_for.squeeze(1))
            ssim = list(map(add, current_ssim, ssim))
            #
            # MMD
            current_mmd = metrics.MMD(x, x_for.squeeze(1))
            mmd = list(map(add, current_mmd.cpu().numpy(), mmd))
            #
            # # PSNR
            current_psnr = metrics.psnr(x, x_for.squeeze(1))
            psnr = list(map(add, current_psnr, psnr))

            # RMSE
            x_new = inv_scaler(stacked_pred, max_value=x_for_unorm.max(), min_value=x_for.min())
            current_rmse = metrics.RMSE(x_new.squeeze(1), x_for_unorm.squeeze(1)) # divide by ten only for geop data
            rmse = list(map(add, current_rmse.cpu().numpy(), rmse))

            # EMD
            # current_emd = []
            # for i in range(args.bsz):
            #    1

            # current_emd = np.array(current_emd)
            # emd = list(map(add, current_emd, emd))
            # print(ssim[0], psnr[0], mmd[0], emd[0])
            # pdb.set_trace()
            print('3 h', current_rmse[3], current_psnr[3], current_ssim[3])#, emd[0])
            print('20 h', current_rmse[20], current_psnr[20], current_ssim[20])#, emd[0])

            print(batch_idx)
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

def metrics_eval_all():

    print("Creating unified plot from text files ...")

    # pdb.set_trace()
    path = os.getcwd() + '/experiments/'

    def read_metrics(fname):

        ssim = []
        psnr = []
        rmse = []
        mmd = []
        lines = []

        # read metric results from file
        print('Reading file:', fname)
        with open(path + fname, 'r') as f:
            line = f.readline()

            while line != '':
                print(line, end='')
                line = f.readline()

                if line == 'Avrg MMD over forecasting period:\n':
                    rmse = lines
                    lines = []

                elif line == '':
                    pass

                else:
                    lines.append(float(line))
            mmd = lines

        return rmse, mmd

    # avrg_psnr_l1k8, avrg_ssim_l1k8 = read_metrics('metric_results_flow-1-level-8-k.txt')
    # avrg_psnr_l2k4, avrg_ssim_l2k4 = read_metrics('metric_results_flow-2-level-4-k.txt')
    # avrg_psnr_l3k4, avrg_ssim_l3k4 = read_metrics('metric_results_flow-3-level-4-k.txt')
    # avrg_psnr_3dunet, avrg_ssim_3dunet = read_metrics('metric_results_3dunet.txt')
    avrg_rmse_3dunet, avrg_mmd_3dunet = read_metrics('metric_results_wbench3dunet_30days.txt')
    # avrg_rmse_l3k4, avrg_mmd_l3k4 = read_metrics('metric_results_30daysera5_flow.txt')
    # avrg_rmse_l1k8, avrg_mmd_l1k8 = read_metrics('metric_results_flow_era5_1l8k.txt')
    avrg_rmse_l3k3, avrg_mmd_l3k3 = read_metrics('metric_results_wbench_l3k3.txt')

    plt.plot(avrg_rmse_l1k8, label='ST-Flow L-1 K-8', color='darkviolet')
    plt.plot(avrg_rmse_l3k4, label='ST-Flow L-3 K-4', color='deeppink')
    plt.plot(avrg_rmse_l3k3, label='ST-Flow L-3 K-3', color='mediumslateblue')
    plt.plot(avrg_rmse_3dunet, label='3DUnet', color='lightseagreen')
    plt.grid(axis='y')
    plt.axvline(x=2, color='orangered')
    plt.legend(loc='best')
    plt.xlabel('Time-Step')
    plt.ylabel('Average RMSE')
    plt.show()
    plt.savefig(path + '/avrg_rmse_all_wbench.png', dpi=300)

    # plt.plot(avrg_ssim_l1k8, label='ST-Flow L1-K-8 Best SSIM', color='darkviolet')
    # plt.plot(avrg_ssim_l3k4, label='ST-Flow L3-K-4 Best SSIM', color='deeppink')
    # plt.plot(avrg_ssim_l2k4, label='ST-Flow L2-K-4 Best SSIM', color='mediumslateblue')
    # plt.plot(avrg_ssim_3dunet, label='3DUnet Best SSIM', color='lightseagreen')
    # plt.grid(axis='y')
    # plt.axvline(x=1, color='orangered')
    # plt.legend(loc='upper right')
    # plt.xlabel('Time-Step')
    # plt.ylabel('Average SSIM')
    # plt.savefig(path + '/avrg_ssim.png', dpi=300)
    # plt.show()

    # plt.plot(avrg_psnr_l1k8, label='ST-Flow L1-K-8 Best PSNR', color='darkviolet')
    # plt.plot(avrg_psnr_l2k4, label='ST-Flow L2-K-4 Best PSNR', color='deeppink')
    # plt.plot(avrg_psnr_l3k4, label='ST-Flow L3-K-4 Best PSNR', color='mediumslateblue')
    # plt.plot(avrg_psnr_3dunet, label='3DUnet Best PNSR', color='lightseagreen')
    # plt.grid(axis='y')
    # plt.axvline(x=1, color='orangered')
    # plt.legend(loc='upper right')
    # plt.xlabel('Time-Step')
    # plt.ylabel('Average PSNR')
    # plt.savefig(path + '/avrg_psnr.png', dpi=300)
    # plt.show()

    return None

if __name__ == "__main__":

    print(torch.cuda.device_count())

    # NOTE: when executing code, make sure you enable the --testmode flag !!!

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    height, width = next(iter(test_loader))[0].shape[3], next(iter(test_loader))[0].shape[4]

    args.device = "cuda"

    # Load Model
    # with downscaling
    # srmodelname = 'model_epoch_1_step_25750'
    # srmodelpath = os.getcwd() + '/experiments/flow-3-level-4-k/models/{}.tar'.format(modelname)
    # stmodelname = 'model_epoch_1_step_25750'
    # stmodelpath = os.getcwd() + '/experiments/flow-3-level-4-k/models/{}.tar'.format(modelname)

    # no downscaling
    modelname = 'model_epoch_0_step_1500'
    modelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/flow_wbench_no_ds__2023_11_28_06_29_14/model_checkpoints/{}.tar'.format(modelname)

    model = condNF.FlowModel((in_channels, height, width),
                              args.filter_size, args.L, args.K, args.bsz,
                              args.lag_len, args.s, args.nb, args.device,
                              args.condch, args.nbits,
                              args.noscale, args.noscaletest, args.testmode)

    print(torch.cuda.device_count())
    pdb.set_trace()
    ckpt = torch.load(modelpath, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)

    print("Evaluate on test split ...")
    test(model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999, args)
    # metrics_eval(args, model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999)
    # metrics_eval_all()
