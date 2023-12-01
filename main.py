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
import os

# Models
from models.architectures import condNF, srflow

# Optimization
from optimization import trainer_stflow, trainer_stflow_ds

import pdb
from tensorboardX import SummaryWriter

import sys
sys.path.append("../../")

###############################################################################


def main(args):
    print(torch.cuda.device_count())
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize device on which to run the model
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.num_gpus = torch.cuda.device_count()
        args.parallel = False

    else:
        args.device = "cpu"

    print("Device", args.device)
    # args.device = "cpu"

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_K{}_L{}_lr{:.4f}_s{}".format(
            args.modeltype, args.trainset, args.bsz, args.Kst, args.Lst, args.lr, args.s
        )

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # train configs
    parser.add_argument("--modeltype", type=str, default="flow",
                        help="Specify modeltype you would like to train [flow, diff, unet3d, convLSTM].")
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
    parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
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
    parser.add_argument("--filter_size", type=int, default=512//2,
                        help="filter size NN in Affine Coupling Layer")
    parser.add_argument("--Lst", type=int, default=3, help="# of levels")
    parser.add_argument("--Kst", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--Lsr", type=int, default=3, help="# of levels")
    parser.add_argument("--Ksr", type=int, default=2,
                        help="# of flow steps, i.e. model depth")
    parser.add_argument("--nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
    parser.add_argument("--condch", type=int, default=128//8,
                        help="# of residual-in-residual blocks in LR network.")

    # data
    parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
                        help="Dataset to train the model on.")
    parser.add_argument("--trainset", type=str, default="wbench",
                        help="Dataset to train the model on.")

    args = parser.parse_args()
    main(args)
