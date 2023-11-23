"""
Taken from https://github.com/roserustowicz/crop-type-mapping/
Implementation by the authors of the paper :
"Semantic Segmentation of crop type in Africa: A novel Dataset and analysis of deep learning methods"
R.M. Rustowicz et al.
Slightly modified to support image sequences of varying length in the same batch.
Adapted more from: https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/unet3d.py
by Christina Winkler, Oktober 2022
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append("../../")

# from data.torch_dataset import ERA5T2MData

def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim, track_running_stats=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim, track_running_stats=True),
        nn.LeakyReLU(inplace=True))
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim, track_running_stats=True),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim, track_running_stats=True),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim, track_running_stats=True),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    def __init__(self, in_channel, hidden_dims=[642,128,256,512,512],
                 pad_value=None, zero_pad=True):

        super(UNet3D, self).__init__()

        self.in_channel = in_channel
        self.pad_value = pad_value
        self.zero_pad = zero_pad

        self.upsample_conv = nn.Conv3d(in_channel, 8, kernel_size=1)

        self.en3 = conv_block(8, hidden_dims[0], hidden_dims[1])
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(hidden_dims[1], hidden_dims[2], hidden_dims[2])
        self.upsample_channel_en4 = nn.Conv3d(1, 3, kernel_size=1)
        self.pool_4 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.center_in = center_in(hidden_dims[2], hidden_dims[3])
        self.center_out = center_out(hidden_dims[3], hidden_dims[2])
        self.pool_center_out = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # TODO: check the upsampling path
        self.downsample_center_out = nn.Conv3d(3, 1, kernel_size=1)
        self.dc4 = conv_block(hidden_dims[3], hidden_dims[2], hidden_dims[2])
        self.trans3 = up_conv_block(hidden_dims[2], hidden_dims[1])
        self.dc3 = conv_block(hidden_dims[2], hidden_dims[2], 1)

        self.downsample_conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)

    def forward(self, x, batch_positions=None):

        x = x.permute(0, 2, 1, 3, 4) # x: [BSZ, C, D, H, W]
        out = self.upsample_conv(x)

        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask

            if self.zero_pad:
                out[out == self.pad_value] = 0

        en3 = self.en3(out)
        # print("en3", en3.shape)
        pool_3 = self.pool_3(en3)
        # print("pool3", pool_3.shape)
        en4 = self.en4(pool_3)
        # print("en4", en4.shape)
        en4 = self.upsample_channel_en4(en4.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        pool_4 = self.pool_4(en4)
        # print("pool4", pool_4.shape)
        center_in = self.center_in(en4)
        # print("center in shape", center_in.shape)
        center_out = self.center_out(center_in)
        # print("center out shape", center_out.shape)
        center_out = self.pool_center_out(center_out)
        # print("center out pooled", center_out.shape)
        center_out = self.downsample_center_out(center_out.permute(0,2,1,3,4))
        center_out = center_out.permute(0,2,1,3,4)
        # print('center out downsampled', center_out.shape)
        concat4 = torch.cat([center_out, en4[:, :, :center_out.shape[2], :, :]], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3[:, :, :trans3.shape[2], :, :]], dim=1)

        # print("concat3", concat3.shape)
        dc3 = self.dc3(concat3)
        # print('dc3', dc3.shape)
        # bring into desired prediction shape
        out = self.downsample_conv(dc3.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        # print("prediction", out.shape)
        return out

# if __name__ == "__main__":
#
#     # test U-net on ERA5 Data
#     # get data
#     datashape = ERA5T2MData('/home/christina/Documents/codecw/spatio-temporal-flow/code/data/assets/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr')[0][0].shape
#     temperatures, time, lat, lon = ERA5T2MData('/home/christina/Documents/codecw/spatio-temporal-flow/code/data/assets/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr')[0]
#     # print("Input shape", temperatures.shape)
#
#     # slice input tensor
#     input = temperatures[:-1,:,:,:]
#
#     # initialize U-Net3D
#     batch_temp = input.unsqueeze(0).float()
#     print(batch_temp.shape)
#     tempUNet = UNet3D(in_channel=1)
#     # print(tempUNet)
#
#     # test forward pass
#     out = tempUNet(batch_temp)
#     print(out.shape)
#
#     # test for bsz 2
#     batch_2 = torch.cat((batch_temp,batch_temp), dim=0)
#     print("batch size 2", batch_2.shape, type(batch_2))
#
#     out2 = tempUNet(batch_2)
