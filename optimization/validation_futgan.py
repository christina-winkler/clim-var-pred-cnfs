import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms

import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb

def validate(generator, discriminator, val_loader, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    color = 'inferno' if args.trainset == 'era5' else 'viridis'

    loss_list=[]
    generator.eval()
    # print(len(val_loader))
    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            x = item[0].to(args.device)

            # split time series into lags and prediction window
            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]
            noise = torch.randn_like(x_past)[:,:,0,...]
            gen_x_for = generator(x_past,noise)
            fake_score = discriminator(gen_x_for)

            # Generative loss
            loss_g = -torch.mean(fake_score)
            loss_list.append(loss_g.mean().detach().cpu().numpy())

            if batch_idx == 100:
                break

            # ---------------------- Evaluate Predictions---------------------- #


        savedir = "{}/snapshots/validationset_{}/".format(exp_name, args.trainset)

        os.makedirs(savedir, exist_ok=True)

        grid_ground_truth = torchvision.utils.make_grid(x_for[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_ground_truth.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "x_t_step_{}_valid.png".format(logstep), dpi=300)

        # visualize past frames the prediction is based on (context)
        grid_past = torchvision.utils.make_grid(x_past[0:9, -1, :, :].cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "_x_t_step_{}_valid.png".format(logstep), dpi=300)

        grid_mu0 = torchvision.utils.make_grid(gen_x_for[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0")
        plt.savefig(savedir + "prediction_logstep_{}_valid.png".format(logstep), dpi=300)

    print("Average Validation Loss:", np.mean(loss_list))
    return np.mean(loss_list)
