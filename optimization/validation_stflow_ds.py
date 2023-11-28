import numpy as np
import torch
import random

import PIL
import os
import torchvision
import torch.nn.functional as F
from torchvision import transforms

import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb

def validate(srmodel, stmodel, val_loader, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    state=None
    nll_list=[]
    srmodel.eval()
    stmodel.eval()

    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            x = item[0].to(args.device)

            x_for, x_past = x[:,:, :1,...].squeeze(1), x[:,:,1:,...]

            x_resh = F.interpolate(x[:,0,...], (16,32)).to(args.device)

            # split time series into lags and prediction window
            x_past_lr, x_for_lr = x_resh[:,:-1,...], x_resh[:,-1,...].unsqueeze(1)

            # reshape into correct format for 3D convolutions - but now i dont use them anymore? xD
            x_past_lr = x_past_lr.unsqueeze(1).contiguous().float().to(args.device)
            x_for_lr = x_for_lr.unsqueeze(1).contiguous().float().to(args.device)

            # run forecasting method
            z, state, nll_st = stmodel.forward(x=x_for_lr, x_past=x_past_lr, state=state)

            # run SR model
            x_for_hat_lr, _ = stmodel._predict(x_past_lr.cuda(), state)
            z, nll_sr = srmodel.forward(x_hr=x_for, xlr=x_for_hat_lr.squeeze(1))

            # Generative loss
            nll_list.append(nll_st.mean().detach().cpu().numpy())

            if batch_idx == 50:
                break

            # ---------------------- Evaluate Predictions---------------------- #

        # evalutae for different temperatures (just for last batch, perhaps change l8er)
        mu0, _ = stmodel._predict(x_past_lr, state, eps=0)
        mu05, _ = stmodel._predict(x_past_lr, state, eps=0.5)
        mu08, _ = stmodel._predict(x_past_lr, state, eps=0.8)
        mu1, _ = stmodel._predict(x_past_lr, state, eps=1)

        # super-resolve
        mu0, _, _ = srmodel(x_hr=x_for, xlr=mu1.squeeze(1), reverse=True, eps=0)
        mu05, _, _ = srmodel(x_hr=x_for, xlr=mu1.squeeze(1), reverse=True, eps=0.5)
        mu08, _, _ = srmodel(x_hr=x_for, xlr=mu1.squeeze(1), reverse=True, eps=0.8)
        mu1, _, _ = srmodel(x_hr=x_for, xlr=mu1.squeeze(1), reverse=True, eps=1.0)
        
        savedir = "{}/snapshots/validationset_{}/".format(exp_name, args.trainset)

        os.makedirs(savedir, exist_ok=True)

        grid_ground_truth = torchvision.utils.make_grid(x_for_lr[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_ground_truth.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "x_t_step_{}_valid.png".format(logstep), dpi=300)

        # visualize past frames the prediction is based on (context)
        grid_past = torchvision.utils.make_grid(x_past_lr[0:9, -1, :, :].cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "_x_t_step_{}_valid.png".format(logstep), dpi=300)

        grid_mu0 = torchvision.utils.make_grid(mu0[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0")
        plt.savefig(savedir + "mu_0_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu05 = torchvision.utils.make_grid(mu05[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.5")
        plt.savefig(savedir + "mu_0.5_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.8")
        plt.savefig(savedir + "mu_0.8_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=1.0")
        plt.savefig(savedir + "mu_1_logstep_{}_valid.png".format(logstep), dpi=300)

        abs_err = torch.abs(mu1 - x_for)
        grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu())
        plt.figure()
        plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Abs Err at t (valid), mu=1.0")
        plt.savefig(savedir + "abs_error_logstep_{}_valid.png".format(logstep), dpi=300)

    print("Average Validation Neg. Log Probability Mass:", np.mean(nll_list))
    return np.mean(nll_list)
