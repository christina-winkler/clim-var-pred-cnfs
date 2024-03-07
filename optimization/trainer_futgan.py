from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json
from math import floor, ceil

# Utils
from utils import utils
import numpy as np
import random
import pdb
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

class ScheduleResolution():
    def __init__(self, args, len_dset, fadein):
        self.config = args
        self.resl = 2
        self.trns_tick = 10
        self.stab_tick = 10
        self.batch_size = args.bsz
        self.len_dset = len_dset
        self.fadein = {'G':None, 'D':None} # TODO adapt
        self.nsamples = 0

    def schedule_resl(self):

        # trns and stab if resl > 2
        if floor(self.resl)!=2:
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick

        # alpha and delta parameters for smooth fade-in (resl-interpolation)
        delta = 1.0/(self.trns_tick+self.stab_tick)
        d_alpha = 1.0*self.batch_size/self.trns_tick/self.len_dset

        # update alpha if FadeInLayer exist
        if self.fadein['D'] is not None:
            if self.resl%1.0 < (self.trns_tick)*delta:
                import pdb; pdb.set_trace()
                self.fadein['G'][0].update_alpha(d_alpha)
                self.fadein['G'][1].update_alpha(d_alpha)
                self.fadein['D'].update_alpha(d_alpha)
                self.complete = self.fadein['D'].alpha*100
                self.phase = 'trns'
            elif self.resl%1.0 >= (self.trns_tick)*delta and self.phase != 'final':
                self.phase = 'stab'

        # increase resl linearly every tick
        prev_nsamples = self.nsamples
        self.nsamples = self.nsamples + self.batch_size
        if (self.nsamples%self.len_dset) < (prev_nsamples%self.len_dset):
            self.nsamples = 0

            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network.
            if self.flag_flush and self.resl%1.0 >= (self.trns_tick)*delta and prev_resl!=2:
                if self.fadein['D'] is not None:
                    self.fadein['G'][0].update_alpha(d_alpha)
                    self.fadein['G'][1].update_alpha(d_alpha)
                    self.fadein['D'].update_alpha(d_alpha)
                    self.complete = self.fadein['D'].alpha*100
                self.flag_flush = False
                self.G.module.flush_network()   # flush G
                self.D.module.flush_network()   # flush and,
                self.fadein['G'] = None
                self.fadein['D'] = None
                self.complete = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'stab'
                self.print_model_structure()

            # grow network.
            if floor(self.resl) != prev_resl and floor(self.resl)<self.max_resl+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.module.grow_network(floor(self.resl))
                self.D.module.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['G'] = [self.G.module.model.fadein_block_decode, self.G.module.model.fadein_block_encode]
                self.fadein['D'] = self.D.module.model.fadein_block
                self.flag_flush = True
                self.print_model_structure()

            if floor(self.resl) >= self.max_resl and self.resl%1.0 >= self.trns_tick*delta:
                self.phase = 'final'
                self.resl = self.max_resl+self.trns_tick*delta

def trainer(args, train_loader, valid_loader, generator, discriminator,
            device='cpu', needs_init=True, ckpt=None):

    config_dict = vars(args)
    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_no_ds_'  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    schedule_resl = ScheduleResolution(args, len(train_loader), {'G': generator, 'D': discriminator})

    criterion = torch.nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, amsgrad=True)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, amsgrad=True)

    paramsG = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    paramsD = sum(x.numel() for x in discriminator.parameters() if x.requires_grad)

    print("Gen:", paramsG)
    print("Disc:", paramsD)

    params = paramsG + paramsD
    print('Nr of Trainable Params on {}:  '.format(device), params)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=2 * 10 ** 5,
    #                                             gamma=0.5)
    if args.resume:
        print('Loading optimizer state dict')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    color = 'inferno' if args.trainset == 'era5' else 'viridis'

    # write training configs to file
    hparams = {'lr': args.lr, 'bsize':args.bsz}

    with open(args.experiment_dir + '/configs.txt','w') as file:
        file.write(json.dumps(hparams))

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            x = item[0].to(device)

            # split time series into lags and prediction window
            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]

            # schedule resolution
            schedule_resl.schedule_resl()

            generator.train()
            discriminator.train()

            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # interpolate discriminator real output
            # TODO self.x.data = self.feed_interpolated_input(self.get_batch())

            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            # if needs_init and torch.cuda.device_count() > 1:
            #     bsz_p_gpu = args.bsz // torch.cuda.device_count()
            #     _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
            #                                 xlr=x[:bsz_p_gpu],
            #                                 logdet=0)

            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)
            writer.add_scalar("nll_train", nll.mean().item(), step)

            # Compute gradients
            nll.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, NLL {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    nll.mean()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    if hasattr(model, "module"):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    model.eval()

                    # testing reconstruction - should be exact same as x_for
                    # pdb.set_trace()
                    reconstructions, _, _ = model.forward(z=z.cuda(), x_past=x_past.cuda(), state=state,
                                      use_stored=True, reverse=True)

                    squared_recon_error = (reconstructions-x_for).mean()**2
                    print("Reconstruction Error:", (reconstructions-x_for).mean())
                    # wandb.log({"Squared Reconstruction Error" : squared_recon_error})

                    grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")
                    # wandb.log({"Reconstructions (train) {}".format(step) : reconstructions})

                    plt.figure()
                    plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(),cmap=color)
                    plt.axis('off')
                    plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
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
                    predictions, _, _ = model._predict(x_past.cuda(), state) # TODO: sample longer trajectories
                    grid_pred = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    future_pred = wandb.Image(cmap_pred, caption="Frame at t")
                    # wandb.log({"Predicted frame at t (train) {}".format(step) : future_pred})

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
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
