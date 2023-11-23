from datetime import datetime
import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim

# Utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from optimization.validation_convlstm_baseline import validate

import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True):

    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    mse = nn.MSELoss()
    prev_mse_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    state=None

    model.to(device)

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            x = item[0].to(device)

            # split time series into lags and prediction window
            x_past, x_for = x[:,:-1,...].float(), x[:,-1,:,:,:].unsqueeze(1).float()

            # reshape into correct format [bsz, num_channels, seq_len, height, width]
            x_past = x_past.permute(0,2,1,3,4).contiguous().float().to(device)
            x_for = x_for.permute(0,2,1,3,4).contiguous().float().to(device)

            model.train()
            optimizer.zero_grad()

            # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
                                            xlr=x[:bsz_p_gpu],
                                            logdet=0)

            out = model.forward(x_past)
            mse_loss = mse(out, x_for)
            writer.add_scalar("mse_loss", mse_loss.item(), step)

            # Compute gradients
            mse_loss.backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, MSE Loss {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    mse_loss))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    if hasattr(model, "module"):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    model.eval()

                    mse_valid = validate(model_without_dataparallel,
                                         valid_loader,
                                         args.experiment_dir,
                                         "{}".format(step),
                                         args,
                                         device=device)

                    writer.add_scalar("mse_valid", mse_valid.mean().item(),
                                       logging_step)

                    # save checkpoint only when validation nll lower than previous model
                    print("Saving Checkpoint !")
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': mse_valid.mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_mse_epoch = mse_valid

                    logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
            print("Done Training for {} mini-batch update steps!".format(args.max_steps)
            )

            if hasattr(model, "module"):
                model_without_dataparallel = model.module
            else:
                model_without_dataparallel = model

            utils.save_model(model_without_dataparallel,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            break
