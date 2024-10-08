from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json

# Utils
from utils import utils
import numpy as np
import random
import pdb
import copy
from tqdm import tqdm
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.architectures.conv_lstm import *
from optimization.validation_stflow import validate

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

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def trainer(args, train_loader, valid_loader, diffusion, model,
            device='cpu', needs_init=True, ckpt=None):

    config_dict = vars(args)
    # wandb.init(project="stdiff", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_no_ds_'  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    mse = nn.MSELoss()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    if args.resume:
        print('Loading optimizer state dict')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    state=None
    color = 'inferno' if args.trainset == 'era5' else 'viridis'

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(args.device), params)

    # write training configs to file
    hparams = {'lr': args.lr, 'bsize':args.bsz, 'Flow Steps':args.Ksr, 'Levels':args.Lsr, 's':args.s, 'ds': args.ds}

    with open(args.experiment_dir + '/configs.txt','w') as file:
        file.write(json.dumps(hparams))

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True


    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        for batch_idx, item in enumerate(pbar):

            x = item[0].to(args.device)

            # split time series into lags and prediction window
            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]

            model.train()
            optimizer.zero_grad()

            t = diffusion.sample_timesteps(x.shape[0]).to(args.device)
            x_past_noisy, noise = diffusion.noise_images(x_past, t)

            predicted_noise = model(x_past_noisy.squeeze(1), t, x_for)

            # compute gradients
            loss = mse(predicted_noise, noise)
            loss.backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, MSE {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    loss.item()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    if hasattr(model, "module"):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    model.eval()

                    sampled_images = diffusion.sample(model, n=1, targets=x_for)
                    ema_sampled_images = diffusion.sample(ema_model, n=1, targets=x_for)

                    # testing reconstruction - should be exact same as x_for
                    # pdb.set_trace()
                    # reconstructions, _, _ = model.forward(z=z.cuda(), x_past=x_past.cuda(), state=state,
                    #                   use_stored=True, reverse=True)
                    #
                    # squared_recon_error = (reconstructions-x_for).mean()**2
                    # print("Reconstruction Error:", (reconstructions-x_for).mean())
                    # wandb.log({"Squared Reconstruction Error" : squared_recon_error})

                    # grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    # array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    # cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    # reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")
                    # wandb.log({"Reconstructions (train) {}".format(step) : reconstructions})
                    # plt.figure()
                    # plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(),cmap=color)
                    # plt.axis('off')
                    # plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
                    # plt.show()

                    # visualize past frames the prediction is based on (context)
                    grid_past = torchvision.utils.make_grid(x_past[0:9, -1, :, :].cpu(), normalize=True, nrow=3)
                    array_imgs_past = np.array(grid_past.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_past = np.apply_along_axis(cm.inferno, 2, array_imgs_past)
                    past_imgs = wandb.Image(cmap_past, caption="Frame at t-1")
                    # wandb.log({"Context Frame at t-1 (train) {}".format(step) : past_imgs})

                    plt.figure()
                    plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Context Frame at t-1 (train)")
                    plt.savefig(viz_dir + '/frame_at_t-1_{}.png'.format(step), dpi=300)

                    # # visualize future frame of the correct prediction
                    grid_future = torchvision.utils.make_grid(x_for[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    array_imgs_future = np.array(grid_future.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_future = np.apply_along_axis(cm.inferno, 2, array_imgs_future)
                    future_imgs = wandb.Image(cmap_future, caption="Frame at t")
                    # wandb.log({"Frame at t (train) {}".format(step) : future_imgs})

                    plt.figure()
                    plt.imshow(grid_future.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Ground Truth at t")
                    plt.savefig(viz_dir + '/frame_at_t_{}.png'.format(step), dpi=300)

                     # predicting a new sample based on context window
                    print("Predicting ...")
                    # predictions, _, _ = model._predict(x_past.cuda(), state) # TODO: sample longer trajectories
                    grid_pred = torchvision.utils.make_grid(sampled_images[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    future_pred = wandb.Image(cmap_pred, caption="Frame at t")
                    # wandb.log({"Predicted frame at t (train) {}".format(step) : future_pred})

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(sampled_images[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_samples.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Prediction at t")
                    plt.savefig(viz_dir + '/samples_{}.png'.format(step), dpi=300)


            if step % args.val_interval == 0:
                print('Validating model ... ')
                nll_valid = validate(model_without_dataparallel,
                                     valid_loader,
                                     args.experiment_dir,
                                     "{}".format(step),
                                     args)

                writer.add_scalar("nll_valid",
                                  nll_valid.mean().item(),
                                  logging_step)

                # save checkpoint only when nll lower than previous model
                if nll_valid < prev_nll_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': nll_valid.mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_nll_epoch = nll_valid

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
            wandb.finish()
            break
