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

from optimization.spatial_utils import make_spates, make_sparse_weight_matrix, temporal_weights
from optimization.gan_utils import original_sinkhorn_loss

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

def trainer(args, train_loader, valid_loader, generator, discriminator_h,
            discriminator_m, device='cpu', needs_init=True, ckpt=None):

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
    criterion = torch.nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, amsgrad=True)
    optimizerD_h = optim.Adam(discriminator_h.parameters(), lr=args.lr, amsgrad=True)
    optimizerD_m = optim.Adam(discriminator_m.parameters(), lr=args.lr, amsgrad=True)

    paramsG = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    paramsD_h = sum(x.numel() for x in discriminator_h.parameters() if x.requires_grad)

    print("Gen:", paramsG)
    print("Disc h:", paramsD_h)

    params = paramsG + 2*paramsD_h
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

    # Calculate spatio-temporal embedding
    embedding_op = 'spate'
    stx_method = 'skw' # Sequential Kulldorff-weighted spatio-temporal expectation
    b = 20 #args.dec_weight
    time_steps = 3
    #b = torch.exp(-torch.arange(1, time_steps).flip(0).float() / b).view(1, 1, -1)
    b = temporal_weights(time_steps,b).to(args.device)
    if stx_method=="kw":
      b = 1 / -torch.log(b[0,0,0]) * time_steps-1 # Get temporal weight b from computed weight tensor of length n
      b = torch.exp(-torch.stack([torch.abs(torch.arange(0, time_steps) - t) for t in range(0,time_steps)]) / b)

    w_sparse = make_sparse_weight_matrix(args.height, args.width).to(args.device)
    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            x = item[0].to(device)

            # split time series into lags and prediction window
            # x_past, x_for = x[:,:, :2,...].to(args.device), x[:,:,2:,...].to(args.device)
            # create real training data for discriminator
            x_past = x[:, :, :2, :, :] #.reshape(args.bsz * 2, time_steps, 1, args.height, args.width).to(device)
            x_for = x[:, :, 2: , :, :] #.reshape(args.bsz * 2, time_steps, 1, args.height, args.width).to(device)

            generator.train()
            discriminator_h.train()
            discriminator_m.train()

            generator.zero_grad()
            optimizerD_h.zero_grad()
            optimizerD_m.zero_grad()

            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            # if needs_init and torch.cuda.device_count() > 1:
            #     bsz_p_gpu = args.bsz // torch.cuda.device_count()
            #     _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
            #                                 xlr=x[:bsz_p_gpu],
            #                                 logdet=0)

            # train generator
            z_height, z_width = (5,5)
            y_dim = 20

            z = torch.randn(args.bsz, time_steps, z_height*z_width).to(args.device)
            y = torch.randn(args.bsz, y_dim).to(args.device)
            z_p = torch.randn(args.bsz, time_steps, z_height*z_width).to(args.device)
            y_p = torch.randn(args.bsz, y_dim).to(args.device)
            real_data = x_past[:args.bsz, ...]
            real_data_p = x_past[args.bsz:,...]
            real_data_emb = x_for[:args.bsz, ...]
            real_data_p_emb = x_for[args.bsz:,...]
            fake_data = generator(z, y)
            fake_data_p = generator(z_p, y_p)

            # create spate embedding
            fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method)
            fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method)

            # create real data embedding
            concat_real = torch.cat((real_data, real_data_emb), dim=2)
            concat_fake = torch.cat((fake_data, fake_data_emb), dim=2)
            import pdb; pdb.set_trace()
            loss_d = original_sinkhorn_loss(real_data, fake_data, 0.8, 100, scale=20)


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
