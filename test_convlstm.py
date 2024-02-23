import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms
import torch.nn as nn
import sys
import timeit
sys.path.append("../../")

from os.path import exists, join
from typing import Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import pdb

from models.architectures import unet3d, conv_lstm_baseline
from data import dataloading
from utils import metrics
from operator import add

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

# train configs
parser.add_argument("--model", type=str, default="convlstm",
                    help="Model you want to train.")
parser.add_argument("--modeltype", type=str, default="convlstm",
                    help="Specify modeltype you would like to train [convlstm].")
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

# runtime configs
parser.add_argument("--visual", action="store_true",
                    help="Visualizing the samples at test time.")
parser.add_argument("--noscaletest", action="store_true",
                    help="Disable scale in coupling layers only at test time.")
parser.add_argument("--noscale", action="store_true",
                    help="Disable scale in coupling layers.")
parser.add_argument("--test", action="store_true",
                    help="Model run on test set.")
parser.add_argument("--train", action="store_true",
                    help="If model should be trained.")
parser.add_argument("--resume_training", action="store_true",
                    help="If training should be resumed.")

# hyperparameters
parser.add_argument("--nbits", type=int, default=8,
                    help="Images converted to n-bit representations.")
parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
parser.add_argument("--crop_size", type=int, default=500,
                    help="Crop size when random cropping is applied.")
parser.add_argument("--patch_size", type=int, default=500,
                    help="Training patch size.")
parser.add_argument("--bsz", type=int, default=1, help="batch size")
parser.add_argument("--seq_len", type=int, default=10,
                    help="Total sequnece length needed from dataloader")
parser.add_argument("--lag_len", type=int, default=2,
                    help="Lag legnth of time-series")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate")
parser.add_argument("--filter_size", type=int, default=512,
                    help="filter size NN in Affine Coupling Layer")
parser.add_argument("--L", type=int, default=1, help="# of levels")
parser.add_argument("--K", type=int, default=1,
                    help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                    help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128,
                    help="# of residual-in-residual blocks in LR network.")

# data
parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
                    help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="temp",
                    help="Dataset to train the model on.")
# parser.add_argument("--testset", type=str, default="wbench",
#                     help="Specify test dataset")
# experiments
parser.add_argument("--exp_name", type=str, default="convlstm_1x",
                    help="Name of the experiment.")

args = parser.parse_args()

def create_rollout(model, init_pred, x_for, x_past, lead_time):

    predictions = []
    predictions.append(init_pred[0,:,:,:,:])
    interm = x_past[0,1,:,:,:].unsqueeze(1).cuda()

    for l in range(lead_time):
        context = torch.cat((predictions[l-1].cuda(), interm.cuda()), 1).unsqueeze(2)
        x = model(context)
        predictions.append(x.squeeze(2))
        # pdb.set_trace()
        interm = x[:,0,:,:,:] # update intermediate state

    stacked_pred = torch.stack(predictions, dim=0)

    # compute absolute error images
    abs_err = torch.abs(stacked_pred.cuda() - x_for[:,:,:,:,:].cuda())

    return stacked_pred, abs_err

def test(model, test_loader, exp_name, logstep, args):
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)
    #
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    state=None
    loss = nn.MSELoss()
    loss_list=[]
    avrg_fwd_time = []
    avrg_bw_time = []

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]

            # split time series into context and prediction window
            x_past, x_for = x[:,:-1,...].float().cuda(), x[:,-1,:,:,:].unsqueeze(1).cuda().float()

            start = timeit.default_timer()
            out = model.forward(x_past)
            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)

            mse_loss = loss(out, x_for)

            # MSE loss
            loss_list.append(mse_loss.mean().detach().cpu().numpy())

            # ---------------------- Evaluate Predictions---------------------- #

            print("Forecast ... ")
            rollout_len = args.bsz-1
            nr_of_rollouts = 1
            pred_multiroll = []
            abs_err_multiroll = []

            stacked_pred1, abs_err1 = create_rollout(model, out, x_for, x_past, rollout_len)
            stacked_pred2, abs_err2 = create_rollout(model, out, x_for, x_past, rollout_len)
            stacked_pred3, abs_err3 = create_rollout(model, out, x_for, x_past, rollout_len)
            stacked_pred4, abs_err4 = create_rollout(model, out, x_for, x_past, rollout_len)

            std = (abs_err1 **2 + abs_err2**2 + abs_err3**2 + abs_err4**2)/4
            stack_pred_multiroll = torch.stack((stacked_pred1,stacked_pred2,stacked_pred3,stacked_pred4), dim=0).squeeze(2)

            stack_pred_multiroll = torch.cat((stack_pred_multiroll, std.squeeze(1).unsqueeze(0)), dim=0)
            stack_abserr_multiroll = torch.stack((abs_err1,abs_err2,abs_err3,abs_err4),dim=0).squeeze(2)


            # stack_pred_multiroll = torch.stack(pred_multiroll, dim=1).squeeze(2)
            # stack_abserr_multiroll = torch.stack(abs_err_multiroll, dim=1).squeeze(2)

            savedir = os.getcwd() + '/experiments/convlstm/snapshots/{}_test'.format(args.trainset)

            os.makedirs(savedir, exist_ok=True)

            # Plot Simulated Rollout Trajectories with Std. starting from same context window
            fig, axes = plt.subplots(nrows=nr_of_rollouts+1, ncols=rollout_len)

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)

            grid1 = torchvision.utils.make_grid(stack_pred_multiroll[0,:,...].cpu(), normalize=True, nrow=1)
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
            plt.close()

            # Plot differences of rollout trajectories
            # from the same rollout different frames
            test_diff0 = stack_pred_multiroll[0,1,:,...]-stack_pred_multiroll[0,2,:,...]
            plt.figure()
            plt.imshow(test_diff0.permute(2, 1, 0).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 0")
            plt.savefig(savedir + "/tesdiff1_logstep_{}_test.png".format(batch_idx), dpi=300)
            plt.close()

            # same frames from different rollout - should be black
            test_diff1 = stack_pred_multiroll[0,1,:,...]-stack_pred_multiroll[2,1,:,...]
            plt.figure()
            plt.imshow(test_diff1.permute(2, 1, 0).cpu().numpy(), cmap=color)
            plt.axis('off')
            plt.title("test diff 1")
            plt.savefig(savedir + "/tesdiff2_logstep_{}_test.png".format(batch_idx), dpi=300)
            # plt.show()
            plt.close()

            grid_ground_truth = torchvision.utils.make_grid(x_for[:,:,0,:,:].cpu(), nrow=1)
            plt.figure()
            plt.imshow(grid_ground_truth.permute(2, 1, 0)[:,:,0].contiguous(), cmap='inferno')
            plt.axis('off')
            plt.title("Frame at t+1")
            plt.savefig(savedir + "/x_t+1_logstep_{}.png".format(batch_idx), dpi=300)
            plt.close()

            stacked_pred, abs_err = create_rollout(model, out, x_for, x_past, rollout_len)

            grid_trajec_preds = torchvision.utils.make_grid(stacked_pred.squeeze(1).cpu(), nrow=1)
            plt.figure()
            plt.imshow(grid_trajec_preds.permute(2, 1, 0)[:,:,0].contiguous(), cmap='inferno')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(savedir + "/rolled_out_traj_test_step_{}.png".format(batch_idx), dpi=300)
            plt.close()

            grid_abs_err = torchvision.utils.make_grid(abs_err1.squeeze(1).cpu(), nrow=1)
            plt.figure()
            plt.imshow(grid_abs_err.permute(2, 1, 0)[:,:,0].contiguous(), cmap='inferno')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(savedir + "/absolute_error_test_step_{}.png".format(batch_idx), dpi=300)
            plt.close()

            # visualize past frames the prediction is based on (context)
            grid_past = torchvision.utils.make_grid(x_past[:, -1, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
            plt.axis('off')
            plt.title("Frame at t")
            plt.savefig(savedir + "/x_t_logstep_{}.png".format(batch_idx), dpi=300)
            plt.close()


    print("Average Test MSE-Loss:", np.mean(loss_list))
    return np.mean(loss_list)

class InverseMinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (0, 1)

    def __call__(self, y):
        x = y * (self.max_value - self.min_value) + self.min_value
        return x

def metrics_eval(model, test_loader, exp_name, modelname, logstep):

    print("Metric evaluation on {}...".format(args.trainset))

    # storing metrics
    # ssim = [0] * args.bsz
    # psnr = [0] * args.bsz
    # mmd = [0] * args.bsz
    # emd = [0] * args.bsz
    norm_rmse = [] 
    rmse = []
    w_rmse = []

    savedir = os.getcwd() + '/experiments/convlstm/'
    os.makedirs(savedir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]

            # split time series into context and prediction window
            x_past, x_for = x[:,:, :2,...].to('cuda'), x[:,:,2:,...].to('cuda')

            # track metric over forecasting period
            print("Forecast ... ")
            lead_time = args.bsz -1
            predictions = []
            past = x_past[0,...].unsqueeze(0).cuda()
            out = model.forward(past)
            predictions.append(out)

            interm = x_past[0,:,1,...].unsqueeze(0).unsqueeze(1)

            for l in range(lead_time):
                context = torch.cat((predictions[l], interm), 1)
                predictions.append(model(context.permute(0,2,1,3,4)))
                interm = predictions[-1][:,0,:,:,:].unsqueeze(1)

            stacked_pred = torch.stack(predictions, dim=0).squeeze(1)

            # # SSIM
            # current_ssim = metrics.ssim(stacked_pred.squeeze(1), x_for.squeeze(1))
            # ssim = list(map(add, current_ssim, ssim))
            #
            # # PSNR
            # current_psnr = metrics.psnr(stacked_pred.squeeze(1), x_for.squeeze(1))
            # psnr = list(map(add, current_psnr, psnr))
            # print(psnr[0], "  ", ssim[0])

            # RMSE
            latitude, longitude = item[3], item[4]
            x_for_new = item[1][:,2:,...].to('cuda')
            if args.trainset == 'temp':
                inv_scaler = InverseMinMaxScaler()
                x_new = inv_scaler(stacked_pred).squeeze(1)

            if args.trainset == 'geop':     
                x_new = stacked_pred * (x_for_new.max() - x_for_new.min()) + x_for_new.min()

                      
            w_rmse.append(metrics.weighted_RMSE(x_new.cpu(), x_for_new.cpu(), latitude, longitude))
            pdb.set_trace()
            print(metrics.RMSE(stacked_pred, x_for).mean(1).detach().cpu().numpy())
            norm_rmse.append(metrics.RMSE(stacked_pred, x_for).mean(1).detach().cpu().numpy())
            rmse.append(metrics.RMSE(x_new, x_for_new.cuda()).mean(1).detach().cpu().numpy())

            if batch_idx == 100:
                print(batch_idx)
                break


        # # compute average SSIM for each temperature map on predicted day t
        # avrg_ssim = list(map(lambda x: x/len(test_loader), ssim))
        # # compute average PSNR for each temperature map on predicted day t
        # avrg_psnr = list(map(lambda x: x/len(test_loader), psnr))
        # pdb.set_trace()
        # avrg_rmse = list(map(lambda x: x/100, rmse)) #len(test_loader), rmse)) TODO improve this too complicated haha
        # avrg_mmd = list(map(lambda x: x/20, mmd)) # len(test_loader), mmd))

        # plt.plot(avrg_ssim, label='3DUnet Best SSIM')
        # plt.grid(axis='y')
        # plt.axvline(x=1, color='brown')
        # plt.legend(loc='upper right')
        # plt.xlabel('Time-Step')
        # plt.ylabel('Average SSIM')
        # plt.savefig(savedir + 'plots/avrg_ssim.png', dpi=300)
        # plt.close()
        #
        # plt.plot(avrg_psnr, label='3DUnet Best PSNR')
        # plt.grid(axis='y')
        # plt.axvline(x=1, color='brown')
        # plt.legend(loc='upper right')
        # plt.xlabel('Time-Step')
        # plt.ylabel('Average PSNR')
        # plt.savefig(savedir + 'plots/avrg_psnr.png', dpi=300)
        # plt.close()

        # plt.plot(avrg_rmse, label='3DUnet RMSE')
        # plt.grid(axis='y')
        # plt.axvline(x=1, color='brown')
        # plt.legend(loc='upper right')
        # plt.xlabel('Time-Step')
        # plt.ylabel('Average RMSE')
        # plt.savefig(savedir + 'plots/avrg_rmse.png', dpi=300)
        # plt.close()

        # plt.plot(avrg_mmd, label='3DUnet MMD')
        # plt.grid(axis='y')
        # plt.axvline(x=1, color='brown')
        # plt.legend(loc='upper right')
        # plt.xlabel('Time-Step')
        # plt.ylabel('Average RMSE')
        # plt.savefig(savedir + 'plots/avrg_mmd.png', dpi=300)
        # plt.close()

        # Write metric results to a file in case to recreate plots
        with open(savedir + 'metric_results.txt','w') as f:
            # f.write('Avrg SSIM over forecasting period:\n')
            # for item in avrg_ssim:
            #     f.write("%f \n" % item)
            #
            # f.write('Avrg PSNR over forecasting period:\n')
            # for item in avrg_psnr:
            #     f.write("%f \n" % item)

            f.write('Avrg RMSE:\n')
            for item in np.mean(rmse, axis=0):
                f.write("%f \n" % item)  

            f.write('STD RMSE:\n')
            for item in np.std(rmse, axis=0):
                f.write("%f \n" % item)

            f.write('Norm Avrg RMSE:\n')
            for item in np.mean(norm_rmse, axis=0):
                f.write("%f \n" % item)  

            f.write('Norm STD RMSE:\n')
            for item in np.std(norm_rmse, axis=0):
                f.write("%f \n" % item)

            # f.write('Avrg MMD over forecasting period:\n')
            # for item in avrg_mmd:
            #     f.write("%f \n" % item)

        return None

if __name__ == "__main__":

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[1]

    # Load Model
    print('Load model ...')

    if args.trainset == 'temp':
        # temperature
        modelname = 'model_epoch_5_step_3700.tar'
        modelpath = '/home/mila/c/christina.winkler/climsim_ds/runs/convlstm_temp_2024_02_12_11_18_45/model_checkpoints/{}'.format(modelname)

    elif args.trainset == 'geop':
        # geopotential
        modelname = 'model_epoch_1_step_31100.tar'
        modelpath = "/home/mila/c/christina.winkler/climsim_ds/runs/convlstm_geop_2024_02_12_11_21_35/model_checkpoints/{}".format(modelname)

    model = conv_lstm_baseline.ConvLSTM(in_channels=in_channels, hidden_channels=4*32, out_channels=1).cuda()
    ckpt = torch.load(modelpath)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format('cuda'), params)
    print("Evaluate ConvLSTM on test split ...")

    # test(model.cuda(), test_loader, "unet3d", -99999, args)
    metrics_eval(model.cuda(),test_loader, "3dunet", modelname, -99999)
