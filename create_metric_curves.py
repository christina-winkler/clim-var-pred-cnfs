import numpy as np
import torch
import random

import PIL
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append("../../")

# seeding only for debugging
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Dataset loading
from data import dataloading
from data.era5_temp_dataset import InverseMinMaxScaler

from os.path import exists, join
import matplotlib.pyplot as plt
from matplotlib import transforms
import argparse
import timeit
import pdb

from models.architectures import condNF, srflow
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage

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
                        help="Interval in which results should be loggeexperiments/d.")
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
# parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
#                         help="Dataset to train the model on.")
parser.add_argument("--datadir", type=str, default="/home/christina/Documents/climsim_ds/data",
                     help="Dataset to train the model on.")
# parser.add_argument("--datadir", type=str, default="/home/mil
parser.add_argument("--trainset", type=str, default="geop",
                        help="Dataset to train the model on.")

args = parser.parse_args()





if __name__ == "__main__":

    print(torch.cuda.device_count())

    # NOTE: when executing code, make sure you enable the --testmode flag !!!

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    height, width = next(iter(test_loader))[0].shape[3], next(iter(test_loader))[0].shape[4]

    args.device = "cuda"

    generate_metric_curves()
