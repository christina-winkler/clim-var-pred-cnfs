from os.path import exists, join
from os import listdir
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import torch
import random
import sys
import pdb
import os
import xarray as xr

from data.era5_temp_dataset import ERA5T2MData
from data.weatherbench_dataset import WeatherBenchData

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import cv2
import os
sys.path.append("../")

from data.imresize_bicubic import imresize

# data utils
def is_train_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"]
    )

def is_test_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ["HR.png", "LR.png", ".jpg", ".png", ".jpeg", ".JPEG"]
    )


class Augment:
    def __init__(self, rot90=True, hflip=True, vflip=True):
        self.rot90 = rot90
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self, img):
        # flip
        hflip = self.hflip and random.random() < 0.5
        vflip = self.vflip and random.random() < 0.5
        rot90 = self.rot90 and random.random() < 0.5

        def _augment(img):
            if rot90:
                img = img.transpose(1, 2)
            if hflip:
                img = img.flip(1)
            if vflip:
                img = img.transpose(1, 2).flip(2)
            return img

        augmented_img = _augment(img)
        return augmented_img


class PILToTensor:
    """
    Convert a ``PIL Image`` to tensor.
    Adapted from:
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    def __init__(self, nbits):
        self.nbits = nbits
        self.nbins = 2 ** self.nbits

    def __call__(self, pic):
        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == "F":
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == "1":
            img = self.nbins * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            img = img.float()
            if self.nbits < 8:
                img = img / (2 ** (8 - self.nbits))
                img = torch.floor(img)

            img = img / self.nbins
            return img
        else:
            return img


class Downsample:
    def __init__(self, s):
        self.s = s

    def __call__(self, in_sample):
        h, w = in_sample.size()[1], in_sample.size()[2]
        out_sample = imresize(in_sample, 1 / self.s)
        out_sample = torch.FloatTensor(out_sample)
        return out_sample


class LoadTrainImages(Dataset):
    def __init__(
        self,
        file_dir,
        input_transform=None,
        target_transform=None,
        padme=False,
        args=None,
    ):
        """
        Args:
            imagedir (str): Path to image directory.
            input_transform (callable, optional): Optional input transform
            to be applied.
            target_transform (callable, optional): Optional target
            transform to be applied.
        """
        super(LoadTrainImages, self).__init__()
        self.padme = padme
        self.patch_size = args.patch_size
        self.s = args.s
        self.image_filenames = []
        for subdir in listdir(file_dir):
            image_dir = join(file_dir, subdir)
            f_names = [
                join(image_dir, x) for x in listdir(image_dir) if is_train_image_file(x)
            ]
            self.image_filenames.extend(f_names)

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_ = Image.open(self.image_filenames[idx])
        target = input_.copy()

        if (input_.mode) != "RGB":
            raise ValueError

        if self.input_transform and self.target_transform:

            input_ = self.input_transform(input_)
            target = self.target_transform(input_)

            if self.padme:
                # Pad the input and target to get the desired patch size
                pad_h = self.patch_size - input_.size()[2]
                pad_w = self.patch_size - input_.size()[2]
                input_pad = (0, pad_w, 0, pad_h)
                tar_pad_h = (self.patch_size // self.s) - target.size()[2]
                tar_pad_w = (self.patch_size // self.s) - target.size()[2]
                tar_pad = (0, tar_pad_h, 0, tar_pad_w)
                input_ = F.pad(input_, input_pad, "constant", 0)
                target = F.pad(target, tar_pad, "constant", 0)

        return input_, target


class LoadTestImages(Dataset):
    def __init__(
        self,
        file_dir,
        input_transform=None,
        target_transform=None,
        args=None,
        padme=False,
    ):
        """
        Args:
            imagedir (str): Path to image directory.
            input_transform (callable, optional): Optional input transform
            to be applied.
            target_transform (callable, optional): Optional target
            transform to be applied.
        """
        super(LoadTestImages, self).__init__()
        self.image_filenames = []
        for subdir in listdir(file_dir):
            image_path = join(file_dir, subdir)
            if is_train_image_file(image_path):
                self.image_filenames.extend([image_path])
            else:
                print("Unknown file extension :(")

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.s = args.s
        self.padme = padme
        self.patch_size = args.patch_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_ = Image.open(self.image_filenames[idx])
        (w, h) = input_.size
        target = input_.copy()

        if (input_.mode) != "RGB":
            input_ = cv2.imread(self.image_filenames[idx])
            input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)
            input_ = Image.fromarray(input_)
            target = input_.copy()

        if self.input_transform and self.target_transform:

            input_ = self.input_transform(input_)
            target = self.target_transform(input_)

            if self.padme:
                # Pad the input and target to get the desired patch size
                pad_h = self.patch_size - input_.size()[1]
                pad_w = self.patch_size - input_.size()[2]
                input_pad = (0, pad_w, 0, pad_h)
                tar_pad_h = (self.patch_size // self.s) - target.size()[2]
                tar_pad_w = (self.patch_size // self.s) - target.size()[1]
                tar_pad = (0, tar_pad_h, 0, tar_pad_w)
                input_ = F.pad(input_, input_pad, "constant", 0)
                target = F.pad(target, tar_pad, "constant", 0)
        return input_, target, (w, h)


############################# TRAINING DATA ####################################
def discretize(sample):
    return (sample * 255).to(torch.int32)

def load_bmnist(args):
    train_loader, val_loader, test_loader = bmnist(root='../data/', batch_size=args.bsz,
                                                   num_workers=4, download=False)

    return train_loader, val_loader, test_loader, args

def load_mnist(args):
    train_loader, val_loader, test_loader = mnist(root='../data/', batch_size=args.bsz,
                                                   num_workers=4, download=False)

    return train_loader, val_loader, test_loader, args

def load_cifar10(args):

    print("Loading Cifar10 ...")

    input_tf = transforms.Compose([transforms.ToTensor()])#, discretize])# PILToTensor(args.nbits)
    target_tf = Downsample(args.s)

    trainset = datasets.CIFAR10(root="./data/", train=True,
                                transform=input_tf,
                                download=True)

    testset = datasets.CIFAR10(root="./data/", train=False,
                               transform=input_tf,
                               download=True)

    n_val_images = 10000
    valid_idcs = np.arange(0, len(trainset), len(trainset) // n_val_images)
    train_idcs = np.setdiff1d(range(len(trainset)), valid_idcs)

    train = torch.utils.data.Subset(trainset, train_idcs)
    valid = torch.utils.data.Subset(trainset, valid_idcs)

    valid_loader = data_utils.DataLoader(valid, args.bsz, shuffle=False,
                                         drop_last=True, num_workers=8)
    train_loader = data_utils.DataLoader(train, args.bsz, shuffle=False,
                                         drop_last=True, num_workers=8)
    test_loader = data_utils.DataLoader(testset, args.bsz, num_workers=8,
                                        shuffle=False, drop_last=True)

    return train_loader, valid_loader, test_loader, args

def load_era5(args):

    print("Loading ERA5 ...")

    dpath = os.getcwd() + '/data/assets/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr'
    dataset = ERA5T2MData(data_path=dpath, window_size=args.lag_len)

    n_train_samples = int(len(dataset) // (1/0.7))
    n_val_samples = int(len(dataset) // (1/0.2))
    n_test_samples = int(len(dataset) // (1/0.1))

    train_idcs = [i for i in range(0, n_train_samples)]
    val_idcs = [i for i in range(0, n_val_samples)]
    test_idcs = [i for i in range(0, n_test_samples)]

    trainset = torch.utils.data.Subset(dataset, train_idcs)
    valset = torch.utils.data.Subset(dataset, val_idcs)
    testset = torch.utils.data.Subset(dataset, test_idcs)

    train_loader = data_utils.DataLoader(trainset, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valset, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(testset, args.bsz, shuffle=False,
                                        drop_last=True)

    return train_loader, val_loader, test_loader, args

def load_weather_bench(args):

    print("Loading Weather Bench ...")

    dpath = os.getcwd() + '/data/geopotential_500/'
    dataset = WeatherBenchData(data_path=dpath, window_size=2)

    n_train_samples = int(len(dataset) // (1/0.85))
    n_val_samples = int(len(dataset) // (1/0.05))
    n_test_samples = int(len(dataset) // (1/0.1))


    train_idcs = [i for i in range(0, n_train_samples)]
    val_idcs = [i for i in range(0, n_val_samples)]
    test_idcs = [i for i in range(0, n_test_samples)]

    trainset = torch.utils.data.Subset(dataset, train_idcs)
    valset = torch.utils.data.Subset(dataset, val_idcs)
    testset = torch.utils.data.Subset(dataset, test_idcs)

    train_loader = data_utils.DataLoader(trainset, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valset, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(testset, args.bsz, shuffle=False,
                                        drop_last=True)

    return train_loader, val_loader, test_loader, args

def load_imagenet32(args):

    print("Loading ImageNet32 ...")

    input_tf = PILToTensor(args.nbits)
    target_tf = Downsample(args.s)

    data = LoadTrainImages(
        file_dir="../{}/imagenet32/train_32x32/".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=False,
    )

    n_val_images = 8
    valid_idcs = np.arange(0, len(data), len(data) // n_val_images)
    train_idcs = np.setdiff1d(range(len(data)), valid_idcs)

    train = torch.utils.data.Subset(data, train_idcs)
    valid = torch.utils.data.Subset(data, valid_idcs)

    test = LoadTrainImages(
        file_dir="../{}/imagenet32/valid_32x32".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=False,
    )

    # pytorch dataloader
    train_loader = data_utils.DataLoader(
        train, args.bsz, shuffle=True, num_workers=8, pin_memory=True
    )
    valid_loader = data_utils.DataLoader(valid, args.bsz, shuffle=False, drop_last=True)
    test_loader = data_utils.DataLoader(test, args.bsz, shuffle=False, drop_last=True)

    args.im_shape_hr = (3, 32, 32)
    args.im_shape_lr = (3, 32 // args.s, 32 // args.s)

    return train_loader, valid_loader, test_loader, args

def load_imagenet64(args):

    print("Loading ImageNet64 ...")
    input_tf = PILToTensor(args.nbits)
    target_tf = Downsample(args.s)

    data = LoadTrainImages(
        file_dir="{}/imagenet64/train_64x64".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=False,
    )

    test = LoadTrainImages(
        file_dir="{}/imagenet64/valid_64x64".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=False,
    )

    n_val_images = 10000
    valid_idcs = np.arange(0, len(data), len(data) // n_val_images)
    train_idcs = np.setdiff1d(range(len(data)), valid_idcs)

    train = torch.utils.data.Subset(data, train_idcs)
    valid = torch.utils.data.Subset(data, valid_idcs)

    # pytorch dataloader
    train_loader = data_utils.DataLoader(
        train, args.bsz, shuffle=True, num_workers=8, pin_memory=True
    )
    valid_loader = data_utils.DataLoader(valid, args.bsz, shuffle=False, drop_last=True)
    test_loader = data_utils.DataLoader(test, args.bsz, shuffle=False, drop_last=True)

    args.im_shape_hr = (3, 64, 64)
    args.im_shape_lr = (3, 64 // args.s, 64 // args.s)

    return train_loader, valid_loader, test_loader, args

def load_full_imagenet(args):

    # can not use PyTorch ImageNet dataloader as it is build for the 2012 ImageNet version
    # Changing .JPEG to .jpeg extensions: find . -name "*.JPEG" -exec bash -c 'mv "$1" "${1%.JPEG}".jpeg' - '{}' \;

    print("Loading full ImageNet2011...")
    input_tf = transforms.Compose(
        [
            transforms.RandomCrop(args.crop_size, pad_if_needed=False),
            PILToTensor(args.nbits),
        ]
    )
    target_tf = Downsample(args.s)

    train = LoadTrainImages(
        file_dir="{}/fullimnet/train".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=True,
    )

    valid = LoadTrainImages(
        file_dir="{}/fullimnet/val".format(args.datadir),
        input_transform=input_tf,
        target_transform=target_tf,
        args=args,
        padme=True,
    )

    # pytorch dataloader
    train_loader = data_utils.DataLoader(train, args.bsz, shuffle=True, num_workers=4)
    valid_loader = data_utils.DataLoader(valid, args.bsz, shuffle=True, num_workers=1)

    args.im_shape_hr = (3, args.crop_size, args.crop_size)
    args.im_shape_lr = (3, args.crop_size // args.s, args.crop_size // args.s)

    print("Length train loader:", len(train_loader))

    return train_loader, valid_loader

def load_data(args):

    if args.trainset == "cifar10":
        return load_cifar10(args)

    elif args.trainset == "imagenet32":
        return load_imagenet32(args)

    elif args.trainset == "imagenet64":
        return load_imagenet64(args)

    elif args.trainset == "fullimnet":
        return load_full_imagenet(args)

    elif args.trainset == "bmnist":
        return load_bmnist(args)

    elif args.trainset == "mnist":
        return load_mnist(args)

    elif args.trainset == "era5":
        return load_era5(args)

    elif args.trainset == "wbench":
        return load_weather_bench(args)

    elif args.trainset == "bair":
        return load_bair(args)

    elif args.trainset == "mov-mnist":
        return load_mov_mnist(args)

    else:
        raise ValueError("Dataset not available. Check for typos!")


# if __name__ == "__main__":
