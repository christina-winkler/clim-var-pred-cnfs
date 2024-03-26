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
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.architectures.conv_lstm import *
from optimization.validation_futgan import validate

# import wandb
# os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainer(args, train_loader, valid_loader, generator, discriminator,
            device='cpu', needs_init=True, ckpt=None):

    config_dict = vars(args)
    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_no_ds_'  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    # writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_loss_epoch = np.inf
    logging_step = 0
    step = 0
    criterion = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, amsgrad=True)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, amsgrad=True)

    paramsG = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    paramsD = sum(x.numel() for x in discriminator.parameters() if x.requires_grad)

    print("Gen:", paramsG)
    print("Disc:", paramsD)

    params = paramsG + paramsD
    print('Nr of Trainable Params on {}:  '.format(device), params)

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
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            x = item[0].to(device)

            # split time series into lags and prediction window
            x_past, x_for = x[:,:, :2,...], x[:,:,2:,...]

            generator.train()
            discriminator.train()

            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # generate future sequence
            noise = torch.randn_like(x_past)[:,:,0,...]
            gen_x_for = generator(x_past) # takes in sequence of past frames to predict sequence of future frames

            # distinguish between real and fake sequences
            # compute score - float
            score_real = discriminator(x_for)
            score_fake = discriminator(gen_x_for.detach())

            # define tensors
            real_labels = Variable(torch.FloatTensor(args.bsz, 1).fill_(1)).to(args.device)
            fake_labels = Variable(torch.FloatTensor(args.bsz, 1).fill_(0)).to(args.device)

            # # wasserstein gradient penalty loss
            loss_d = torch.mean(score_fake) - torch.mean(score_real)
            
            # gradient penalty
            lam = 10
            alpha = torch.randn(args.bsz, 1)
            alpha = alpha.expand(args.bsz, x[:,:,2:,:,:][0].nelement()).contiguous().view(args.bsz, x.size(1), x[:,:,2:,:,:].size(2), x.size(3), x.size(4)).to(args.device)
            interpolates = alpha*x[:,:,2:,:,:]+((1-alpha)*gen_x_for).to(args.device)
            interpolates = Variable(interpolates, requires_grad=True).to(args.device)
            interpolates_score = discriminator(interpolates)
            gradients = torch.autograd.grad(outputs=interpolates_score, inputs=interpolates,
                                            grad_outputs=torch.ones(interpolates_score.size()).cuda(),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]

            # compute gradients
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
            # loss_d = loss_d+lam*gradient_penalty

            eps = 0.001
            eps_penalty = torch.mean((score_real-0)**2)
            loss_d = loss_d+eps_penalty*eps

            real_loss = criterion(score_real, real_labels)
            fake_loss = criterion(score_fake, fake_labels)
            
            loss_d = (real_loss + fake_loss) + lam*gradient_penalty
            loss_d.backward()

            # update discriminator
            optimizerD.step()

            # compute adversarial loss
            adv_loss = criterion(discriminator(gen_x_for), real_labels)
            loss_g = mse_loss(gen_x_for, x_for)  #+ 0.01 * adv_loss
            loss_g.backward()
            optimizerG.step()

            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, Disc Loss {:.3f}, Gen Loss {:.3f} ".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    loss_d,
                    loss_g))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    # if hasattr(model, "module"):
                    #     model_without_dataparallel = model.module
                    # else:
                    #     model_without_dataparallel = model

                    generator.eval()

                    # testing reconstruction - should be exact same as x_for
                    # pdb.set_trace()
                    # reconstructions, _, _ = model.forward(z=z.cuda(), x_past=x_past.cuda(), state=state,
                    #                   use_stored=True, reverse=True)
                    #
                    # squared_recon_error = (reconstructions-x_for).mean()**2
                    # print("Reconstruction Error:", (reconstructions-x_for).mean())
                    # # wandb.log({"Squared Reconstruction Error" : squared_recon_error})
                    #
                    # grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    # array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    # cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    # reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")
                    # wandb.log({"Reconstructions (train) {}".format(step) : reconstructions})
                    #
                    # plt.figure()
                    # plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(),cmap=color)
                    # plt.axis('off')
                    # plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
                    # # plt.show()

                    # visualize past frames the prediction is based on (context)
                    # grid_past = torchvision.utils.make_grid(x_past[0:9, -1, :, :].cpu(), normalize=True, nrow=3)
                    # array_imgs_past = np.array(grid_past.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    # cmap_past = np.apply_along_axis(cm.inferno, 2, array_imgs_past)
                    # past_imgs = wandb.Image(cmap_past, caption="Frame at t-1")
                    # wandb.log({"Context Frame at t-1 (train) {}".format(step) : past_imgs})

                    # plt.figure()
                    # plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    # plt.axis('off')
                    # plt.title("Context Frame at t-1 (train)")
                    # plt.savefig(viz_dir + '/frame_at_t-1_{}.png'.format(step), dpi=300)

                    # # visualize future frame of the correct prediction
                    # grid_future = torchvision.utils.make_grid(x_for[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    # array_imgs_future = np.array(grid_future.permute(2,1,0)[:,:,0].unsqueeze(2))
                    # cmap_future = np.apply_along_axis(cm.inferno, 2, array_imgs_future)
                    # future_imgs = wandb.Image(cmap_future, caption="Frame at t")
                    # wandb.log({"Frame at t (train) {}".format(step) : future_imgs})

                    # plt.figure()
                    # plt.imshow(grid_future.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    # plt.axis('off')
                    # plt.title("Ground Truth at t")
                    # plt.savefig(viz_dir + '/frame_at_t_{}.png'.format(step), dpi=300)

                     # predicting a new sample based on context window
                    print("Predicting ...")
                    grid_pred = torchvision.utils.make_grid(gen_x_for[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    # array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    # cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    # future_pred = wandb.Image(cmap_pred, caption="Frame at t")
                    # wandb.log({"Predicted frame at t (train) {}".format(step) : future_pred})

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(gen_x_for[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_pred.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Prediction at t")
                    # plt.show()
                    plt.savefig(viz_dir + '/predictions_{}.png'.format(step), dpi=300)
                    plt.close()

                    # visualize abs error 
                    abs_err = torch.abs(x_for - gen_x_for)
                    grid_abs_err = torchvision.utils.make_grid(abs_err[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_abs_err.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Abs Err at t")
                    # plt.show()
                    plt.savefig(viz_dir + '/abs_err_{}.png'.format(step), dpi=300)
                    plt.close()


            if step % args.val_interval == 0:
                print('Validating model ... ')
                loss_valid = validate(generator, discriminator,
                                      valid_loader,
                                      args.experiment_dir,
                                      "{}".format(step),
                                      args)

                # writer.add_scalar("loss_valid",
                #                   loss_valid.mean().item(),
                #                   logging_step)

                # save checkpoint only when nll lower than previous model
                if loss_valid < prev_loss_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': generator.state_dict(),
                                'optimizer_state_dict': optimizerG.state_dict(),
                                'loss': loss_valid.mean()}, PATH+ f"generator_epoch_{epoch}_step_{step}.tar")
                    prev_loss_epoch = loss_valid

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

            utils.save_model(generator,
                             epoch, optimizerG, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break
