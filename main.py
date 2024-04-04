import sys
sys.path.append("../../")

import argparse
import torch

# Dataset loading
from data import dataloading

# Utils
import utils
import random
import numpy as np
import time
import os

# Models
from models.architectures import condNF, srflow, unet3d, conv_lstm_baseline, future_gan, spate_gan, ddpm_conditional, conv_lstm_diff, diff_modules, threedgan

# Optimization
from optimization import trainer_stflow, trainer_stflow_ds, trainer_stdiff, trainer_stdiff_ds, trainer_unet3d, trainer_convlstm, trainer_futgan, trainer_spategan, trainer_3dgan

import pdb
from tensorboardX import SummaryWriter

import sys
sys.path.append("../../")

###############################################################################


def main(args):

    print(torch.cuda.device_count())
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Initialize device on which to run the model
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.num_gpus = torch.cuda.device_count()
        args.parallel = False

    else:
        args.device = "cpu"

    print("Device", args.device)

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_K{}_L{}_lr{:.4f}_s{}".format(
            args.modeltype, args.trainset, args.bsz, args.Kst,
            args.Lst, args.lr, args.s)

    if args.train:
        # load data
        train_loader, valid_loader, test_loader, args = dataloading.load_data(args)
        in_channels = 1 # next(iter(train_loader))[0].shape[2]

    print("Start training {} on {}:".format(args.modeltype, args.trainset))

    if args.modeltype == "flow":

        height, width = next(iter(train_loader))[0].shape[3], next(iter(train_loader))[0].shape[4]

        ckpt= None
        if args.resume:
            print("Resume training of model ...")
            modelname = 'model_epoch_0_step_7250_wbench.tar'
            # modelpath = os.getcwd() + "/experiments/flow-3-level-3-k/models/{}".format(modelname)
            modelpath = '/home/christina/Documents/spatio-temporal-conditioned-normalizing-flow/code/flow-3-level-3-k_model_epoch_0_step_7250_wbench/model/{}'.format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        if args.ds:

            sr_model = srflow.SRFlow((in_channels, height, width), args.filter_size, args.Lsr, args.Ksr,
                                      args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

            st_model = condNF.FlowModel((in_channels, height//args.s, width//args.s),
                                        args.filter_size, args.Lst, args.Kst, args.bsz,
                                        args.lag_len, args.s, args.nb, args.device,
                                        args.condch, args.nbits,
                                        args.noscale, args.noscaletest).to(args.device)

            trainer_stflow_ds.trainer(args=args, train_loader=train_loader,
                                      valid_loader=valid_loader,
                                      srmodel=sr_model,
                                      stmodel=st_model,
                                      device=args.device,
                                      ckpt=ckpt)

        else:
            print("No correction")
            model = condNF.FlowModel((in_channels, height, width),
                                        args.filter_size, args.Lst, args.Kst, args.bsz,
                                        args.lag_len, args.s, args.nb, args.device,
                                        args.condch, args.nbits,
                                        args.noscale, args.noscaletest).to(args.device)

            trainer_stflow.trainer(args=args, train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    model=model,
                                    device=args.device,
                                    ckpt=ckpt)
    elif args.modeltype == "diff":

        height, width = next(iter(train_loader))[0].shape[3], next(iter(train_loader))[0].shape[4]

        ckpt= None
        if args.resume:
            print("Resume training of model ...")
            modelname = 'model_epoch_0_step_7250_wbench.tar'
            # modelpath = os.getcwd() + "/experiments/flow-3-level-3-k/models/{}".format(modelname)
            modelpath = '/home/christina/Documents/spatio-temporal-conditioned-normalizing-flow/code/flow-3-level-3-k_model_epoch_0_step_7250_wbench/model/{}'.format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        if args.ds or args.s > 1: # with simulation correction or upsampling in the end

            sr_model = srflow.SRFlow((in_channels, height, width), args.filter_size, args.Lsr, args.Ksr,
                                      args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

            # diffusion process model
            diffusion = ddpm_conditional.Diffusion(img_size=(height,width),device=args.device)

            # Gated Conv LSTM for generating the latent representatios
            # convlstm = conv_lstm_diff.ConvLSTM(in_channels=args.lag_len, hidden_channels=32,
            #                               out_channels=1).to(args.device)
            unet = diff_modules.UNet_conditional(c_in=1, c_out=1, time_dim=256,
                                                 num_classes=None, device=args.device)

            trainer_stdiff.trainer(args=args, train_loader=train_loader,
                                   valid_loader=valid_loader,
                                   diffusion=diffusion,
                                   model=unet,
                                   device=args.device,
                                   ckpt=ckpt)

        else: # no upampling, simulation is run on input dimensionality without compression
            print("No compression.")

            # diffusion process model
            diffusion = ddpm_conditional.Diffusion(img_size=(height,width), device=args.device)

            # Conditional 3DUNet
            unet = diff_modules.UNet_conditional(c_in=2, c_out=1, time_dim=256,
                                                 num_classes=height*width, device=args.device).to(args.device)

            trainer_stdiff.trainer(args=args, train_loader=train_loader,
                                   valid_loader=valid_loader,
                                   diffusion=diffusion,
                                   model=unet,
                                   device=args.device,
                                   ckpt=ckpt)


    elif args.modeltype == 'futgan':
        height, width = next(iter(train_loader))[0].shape[3], next(iter(train_loader))[0].shape[4]

        generator = future_gan.FutureGenerator(config=args).to(args.device)
        discriminator = future_gan.Discriminator(config=args).to(args.device)

        print('Training FutureGAN ...')
        trainer_futgan.trainer(args=args, train_loader=train_loader,
                               valid_loader=valid_loader, generator=generator,
                               discriminator=discriminator,
                               device=args.device)

    elif args.modeltype == '3dgan':
        height, width = next(iter(train_loader))[0].shape[3], next(iter(train_loader))[0].shape[4]
        generator = threedgan.Generator(in_c=args.lag_len, out_c=1, height=height, width=width).to(args.device)
        discriminator = threedgan.Discriminator(in_c=1, out_c=1, height=height, width=width).to(args.device)

        print('Training 3DGAN ...')
        trainer_3dgan.trainer(args=args, train_loader=train_loader,
                             valid_loader=valid_loader, generator=generator,
                             discriminator=discriminator,
                             device=args.device)

    elif args.modeltype == 'spategan':
        args.height, args.width = next(iter(train_loader))[0].shape[3], next(iter(train_loader))[0].shape[4]

        generator = spate_gan.VideoDCG(args.bsz, time_steps=3, x_h=args.height, x_w=args.width, filter_size=32,
                                       state_size=32, bn=True, output_act='sigmoid', nchannel=1).to(args.device)
        discriminator_h = spate_gan.VideoDCD(args.bsz, x_h=args.height, x_w=args.width, filter_size=32, j=16,
                                             nchannel=1, bn=True).to(args.device)
        discriminator_m = spate_gan.VideoDCD(args.bsz, x_h=args.height, x_w=args.width, filter_size=32, j=16,
                                             nchannel=1, bn=True).to(args.device)

        print('Training SpateGAN ...')
        trainer_spategan.trainer(args=args, train_loader=train_loader,
                                 valid_loader=valid_loader, generator=generator,
                                 discriminator_h=discriminator_h,
                                 discriminator_m=discriminator_m,
                                 device=args.device)

    elif args.modeltype == "unet3d":
        print('Training 3DUNet!')
        model = unet3d.UNet3D(in_channel=in_channels).to(args.device)

        if args.resume:
            modelname = 'model_epoch_1_step_25700_wbench.tar'
            modelpath = os.getcwd() + "/runs/unet3d_wbench_2023_08_28_11_50_43/model_checkpoints/{}".format(modelname)
            model = unet3d.UNet3D(in_channels)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_unet3d.trainer(args=args, train_loader=train_loader,
                               valid_loader=valid_loader,
                               model=model.cuda(),
                               device=args.device)

    elif args.modeltype == "convlstm":
        print('Training ConvLSTM!')
        model = conv_lstm_baseline.ConvLSTM(in_channels=in_channels, hidden_channels=4*32, out_channels=1).to(args.device)

        if args.resume:
            modelname = 'model_epoch_1_step_25700_wbench.tar'
            modelpath = os.getcwd() + "/runs/unet3d_wbench_2023_08_28_11_50_43/model_checkpoints/{}".format(modelname)
            model = conv_lstm.ConvLSTM(in_channels)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_convlstm.trainer(args=args, train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 model=model.cuda(),
                                 device=args.device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # train configs
    parser.add_argument("--modeltype", type=str, default="flow",
                        help="Specify modeltype you would like to train [flow, diff, unet3d, convLSTM, futgan, spategan].")
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
    parser.add_argument("--s", type=int, default=1, help="Upscaling factor.")
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
    parser.add_argument("--filter_size", type=int, default=16,
                        help="filter size NN in Affine Coupling Layer")
    parser.add_argument("--Lst", type=int, default=3, help="# of levels")
    parser.add_argument("--Kst", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--Lsr", type=int, default=3, help="# of levels")
    parser.add_argument("--Ksr", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--nb", type=int, default=64,
                        help="# of residual-in-residual blocks LR network.")
    parser.add_argument("--condch", type=int, default=64,
                        help="# of residual-in-residual blocks in LR network.")

    # data
    parser.add_argument("--datadir", type=str, default="/home/christina/Documents/climsim_ds/data",
                        help="Dataset to train the model on.")
    # parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
    #                     help="Dataset to train the model on.")
    parser.add_argument("--trainset", type=str, default="geop",
                        help="Dataset to train the model on [geop, temp].")

    # FutureGAN config options
    parser.add_argument('--dgx', type=bool, default=False, help='set to True, if code is run on dgx, default=`False`')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus for (multi-)gpu training, default=1')
    parser.add_argument('--random_seed', type=int, default=int(time.time()), help='seed for generating random numbers, default = `int(time.time())`')
    parser.add_argument('--ext', action='append', default=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'], help='list of strings of allowed file extensions, default=[`.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`, `.pgm`]')
    parser.add_argument('--use_ckpt', type=bool, default=False, help='continue training from checkpoint, default=`False`')

    parser.add_argument('--ckpt_path', action='append', help='list of path(s) to training checkpoints to continue training or for testing, [0] Generator and [1] Discriminator, default=``')
    parser.add_argument('--data_root', type=str, default='', help='path to root directory of training data (ex. -->path_to_dataset/train)')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to directory of log files')
    parser.add_argument('--experiment_name', type=str, default='', help='name of experiment (if empty, current date and time will be used), default=``')

    parser.add_argument('--d_cond', type=bool, default=True, help='condition discriminator on input frames, default=`True`')
    parser.add_argument('--nc', type=int, default=1, help='number of input image color channels, default=3')
    parser.add_argument('--max_resl', type=int, default=128, help='max. frame resolution --> image size: max_resl x max_resl , default=128')
    parser.add_argument('--nframes_in', type=int, default=2, help='number of input video frames in one sample, default=12')
    parser.add_argument('--nframes_pred', type=int, default=1, help='number of video frames to predict in one sample, default=6')
    # p100
    parser.add_argument('--batch_size_table', type=dict, default={4:32, 8:16, 16:8, 32:4, 64:2, 128:1, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
    ## dgx
    #parser.add_argument('--batch_size_table', type=dict, default={4:256, 8:128, 16:64, 32:32, 64:16, 128:8, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
    parser.add_argument('--trns_tick', type=int, default=10, help='number of epochs for transition phase, default=10')
    parser.add_argument('--stab_tick', type=int, default=10, help='number of epochs for stabilization phase, default=10')

    # training
    parser.add_argument('--nz', type=int, default=512, help='dimension of input noise vector z, default=512')
    parser.add_argument('--ngf', type=int, default=512, help='feature dimension of final layer of generator, default=512')
    parser.add_argument('--ndf', type=int, default=512, help='feature dimension of first layer of discriminator, default=512')

    parser.add_argument('--loss', type=str, default='wgan_gp', help='which loss functions to use (choices: `gan`, `lsgan` or `wgan_gp`), default=`wgan_gp`')
    parser.add_argument('--d_eps_penalty', type=bool, default=True, help='adding an epsilon penalty term to wgan_gp loss to prevent loss drift (eps=0.001), default=True')
    parser.add_argument('--acgan', type=bool, default=False, help='adding a label penalty term to wgan_gp loss --> makes GAN conditioned on classification labels of dataset, default=False')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type, default=adam')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('--lr_decay', type=float, default=0.87, help='learning rate decay at every resolution transition, default=0.87')

    parser.add_argument('--lrelu', type=bool, default=True, help='use leaky relu instead of relu, default=True')
    parser.add_argument('--padding', type=str, default='zero', help='which padding to use (choices: `zero`, `replication`), default=`zero`')
    parser.add_argument('--w_norm', type=bool, default=True, help='use weight scaling, default=True')
    parser.add_argument('--batch_norm', type=bool, default=False, help='use batch-normalization (not recommended), default=False')
    parser.add_argument('--g_pixelwise_norm', type=bool, default=True, help='use pixelwise normalization for generator, default=True')
    parser.add_argument('--d_gdrop', type=bool, default=False, help='use generalized dropout layer (inject multiplicative Gaussian noise) for discriminator when using LSGAN loss, default=False')
    parser.add_argument('--g_tanh', type=bool, default=False, help='use tanh at the end of generator, default=False')
    parser.add_argument('--d_sigmoid', type=bool, default=False, help='use sigmoid at the end of discriminator, default=False')
    parser.add_argument('--x_add_noise', type=bool, default=False, help='add noise to the real image(x) when using LSGAN loss, default=False')
    parser.add_argument('--z_pixelwise_norm', type=bool, default=False, help='if mode=`gen`: pixelwise normalization of latent vector (z), default=False')

    # display and save
    parser.add_argument('--tb_logging', type=bool, default=False, help='enable tensorboard visualization, default=True')
    parser.add_argument('--update_tb_every', type=int, default=100, help='display progress every specified iteration, default=100')
    parser.add_argument('--save_img_every', type=int, default=100, help='save images every specified iteration, default=100')
    parser.add_argument('--save_ckpt_every', type=int, default=5, help='save checkpoints every specified epoch, default=5')

    # parse and save training config
    args = parser.parse_args()

    main(args)
