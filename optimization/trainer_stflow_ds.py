from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json 

# utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.architectures.conv_lstm import *
from optimization.validation_stflow_ds import validate
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

import wandb
os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainer(args, train_loader, valid_loader, srmodel, stmodel,
            device='cpu', needs_init=True, ckpt=None):

    config_dict = vars(args)
    # wandb.init(project="arflow", config=config_dict)
    print("With DOWNSCALING")
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_{}x'.format(args.s) + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0

    sr_optimizer = optim.Adam(srmodel.parameters(), lr=args.lr, amsgrad=True)
    st_optimizer = optim.Adam(stmodel.parameters(), lr=args.lr, amsgrad=True)    

    sr_scheduler = torch.optim.lr_scheduler.StepLR(sr_optimizer,
                                                   step_size=2 * 10 ** 5,
                                                   gamma=0.5)

    st_scheduler = torch.optim.lr_scheduler.StepLR(st_optimizer,
                                                   step_size=2 * 10 ** 5,
                                                   gamma=0.5)
    if args.resume:
        print('Loading optimizer state dict')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    state = None
    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    srmodel.to(device)
    stmodel.to(device)

    params = sum(x.numel() for x in srmodel.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)

    # write training configs to file
    hparams = {'lr': args.lr, 'bsize':args.bsz, 'Flow Steps SR':args.Ksr, 'Levels SR':args.Lsr, 
                'Flow Steps ST': args.Kst, 'Levels ST':args.Lst,'s':args.s, 'ds': args.ds}
    
    with open(args.experiment_dir + '/configs.txt','w') as file:
        file.write(json.dumps(hparams))
    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        srmodel = torch.nn.DataParallel(srmodel)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            x = item[0].to(args.device)

            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]

            x_resh = F.interpolate(x[:,0,...], (x_for.shape[3]//args.s,x_for.shape[4]//args.s))

            # split time series into lags and prediction window
            x_past_lr, x_for_lr =  x_resh[:,:2, ...].squeeze(1), x_resh[:,2:,...]

            # reshape into correct format [bsz, num_channels, seq_len, height, width]
            x_past_lr = x_past_lr.unsqueeze(1).contiguous().float()
            x_for_lr = x_for_lr.unsqueeze(1).contiguous().float()

            # print(x_past.shape, x_for.shape)
            srmodel.train()
            stmodel.train()

            sr_optimizer.zero_grad()
            st_optimizer.zero_grad()

            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers. TODO: adapt this
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
                                            xlr=x[:bsz_p_gpu],
                                            logdet=0)

            # run forecasting method
            z, state, nll_st = stmodel.forward(x=x_for_lr, x_past=x_past_lr, state=state)

            # run SR model
            x_for_hat_lr, _ = stmodel._predict(x_past_lr.cuda(), state)
            z, nll_sr = srmodel.forward(x_hr=x_for.squeeze(1), xlr=x_for_hat_lr.squeeze(1))

            # Compute gradients
            nll_sr.mean().backward()
            nll_st.mean().backward()

            # Update model parameters using calculated gradients
            st_optimizer.step()
            sr_optimizer.step()

            st_scheduler.step()
            sr_scheduler.step()

            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, NLL SR {:.3f}, NLL ST {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    nll_sr.mean(),
                    nll_st.mean()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    if hasattr(srmodel, "module"):
                        srmodel_without_dataparallel = srmodel.module
                        stmodel_without_dataparallel = stmodel.module
                    else:
                        srmodel_without_dataparallel = srmodel
                        stmodel_without_dataparallel = stmodel

                    srmodel.eval()
                    stmodel.eval()

                    # testing reconstruction - should be exact same as x_for
                    reconstructions,_,  _ = stmodel.forward(z=z.cuda(), x_past=x_past_lr.cuda(), state=state,
                                                         use_stored=True, reverse=True)

                    squared_recon_error = (reconstructions-x_for_lr).mean()**2
                    print("Reconstruction Error:", (reconstructions-x_for_lr).mean())
                    # wandb.log({"Squared Reconstruction Error" : squared_recon_error})

                    grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")

                    plt.figure()
                    plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(),cmap=color)
                    plt.axis('off')
                    plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
                    # plt.show()

                    # # visualize past frames the prediction is based on (context)
                    # grid_past = torchvision.utils.make_grid(x_past_lr[0:9, -1, :, :].cpu(), normalize=True, nrow=3)
                    # array_imgs_past = np.array(grid_past.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    # cmap_past = np.apply_along_axis(cm.inferno, 2, array_imgs_past)
                    # past_imgs = wandb.Image(cmap_past, caption="Frame at t-1")
                    # # wandb.log({"Context Frame at t-1 (train) {}".format(step) : past_imgs})

                    # plt.figure()
                    # plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    # plt.axis('off')
                    # plt.title("Context Frame at t-1 (train)")
                    # plt.savefig(viz_dir + '/frame_at_t-1_{}.png'.format(step), dpi=300)
                    
                    # # visualize future frame of the correct prediction
                    # grid_future = torchvision.utils.make_grid(x_for_lr[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    # array_imgs_future = np.array(grid_future.permute(2,1,0)[:,:,0].unsqueeze(2))
                    # cmap_future = np.apply_along_axis(cm.inferno, 2, array_imgs_future)
                    # future_imgs = wandb.Image(cmap_future, caption="Frame at t")
                    # # wandb.log({"Frame at t (train) {}".format(step) : future_imgs})

                    # plt.figure()
                    # plt.imshow(grid_future.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    # plt.axis('off')
                    # plt.title("Ground Truth at t")
                    # plt.savefig(viz_dir + '/frame_at_t_{}.png'.format(step), dpi=300)

                     # predicting a new sample based on context window
                    print("Predicting ...")
                    predictions, _ = stmodel._predict(x_past_lr.cuda(), state) # TODO: sample longer trajectories
                    grid_pred = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    future_pred = wandb.Image(cmap_pred, caption="Frame at t")

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_samples.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Low-Res Prediction at t")
                    plt.savefig(viz_dir + '/samples_{}.png'.format(step), dpi=300)

            if step % args.val_interval == 0:

                print('Validating model ... ')

                nll_valid = validate(srmodel_without_dataparallel,
                                     stmodel_without_dataparallel,
                                     valid_loader,
                                     args.experiment_dir,
                                     "{}".format(step),
                                     args)

                writer.add_scalar("nll_valid",
                                  nll_valid.mean().item(),
                                  logging_step)

                # save checkpoint only when nll lower than previous model
                if nll_valid < prev_nll_epoch:

                    SRPATH = args.experiment_dir + '/srmodel_checkpoints/'
                    STPATH = args.experiment_dir + '/stmodel_checkpoints/'

                    os.makedirs(SRPATH, exist_ok=True)
                    os.makedirs(STPATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': srmodel.state_dict(),
                                'optimizer_state_dict': sr_optimizer.state_dict(),
                                'loss': nll_valid.mean()}, SRPATH+ f"model_epoch_{epoch}_step_{step}.tar")

                    torch.save({'epoch': epoch,
                                'model_state_dict': stmodel.state_dict(),
                                'optimizer_state_dict': st_optimizer.state_dict(),
                                'loss': nll_valid.mean()}, STPATH+ f"model_epoch_{epoch}_step_{step}.tar")    

                    prev_nll_epoch = nll_valid

            logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
            print("Done Training for {} mini-batch update steps!".format(args.max_steps)
            )

            if hasattr(srmodel, "module"): # TODO this codeblock likely only works for SR model
                model_without_dataparallel = srmodel.module
            else:
                model_without_dataparallel = sr_modelmodel

            utils.save_model(model_without_dataparallel,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break