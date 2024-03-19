import torch
import torch.nn as nn
import torch.nn.functional as F
from models.architectures import unet3d

"""
Implements model class of our 3D GAN model.
"""

class Generator(torch.nn.Module):
    def __init__(self, in_c, out_c, height, width):
        super(Generator, self).__init__()
        self.bias =  False # params.bias
        self.z_dim = 200 # params.z_dim
        self.f_dim = 64



        padd = (1, 1, 1)
        # note +1 as we only concatenate one noise layer
        self.layer1 = self.conv_layer(3, 1, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        self.unet3d_1 = UNet3D(in_channel=1)
        # self.layer2 = self.conv_layer(self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        # self.attn1 = SelfAttention(2*self.f_dim, height, width)
        # self.layer3 = self.conv_layer(2*self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        # self.layer4 = self.conv_layer(2*self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        # self.attn2 = SelfAttention(self.f_dim, height, width)
        # self.layer5 = self.conv_layer(self.f_dim, 1, kernel_size=3, stride=1, padding=padd, bias=self.bias)


    def conv_layer(self, input_dim, output_dim, kernel_size=3, stride=2, padding=(0,0,0), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            # torch.nn.ReLU(True)
            torch.nn.LeakyReLU(0.4, True)
        )
        return layer

    def forward(self, x, noise):
        out = torch.cat((x, noise.unsqueeze(1).contiguous()), 2).permute(0,2,1,3,4).contiguous()
        out1 = self.layer1(out)
        out2 = self.unet3d_1(out)
        # out = self.layer2(out)
        # out = self.attn1(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.attn2(out)
        # out = self.layer5(out)
        # import pdb; pdb.set_trace()
        return out2 * 0.4 + out1


class Discriminator(torch.nn.Module):
    def __init__(self, in_c, out_c, height, width):
        super(Discriminator, self).__init__()

        self.leak_value = 0.2
        self.bias = False
        self.height = height
        self.width = width

        padd = (1,1,1)
        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)
        # self.attn1 = SelfAttention(2*self.f_dim, height, width)
        self.layer3 = self.conv_layer(2*self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)
        # self.attn2 = SelfAttention(self.f_dim, height, width)
        self.layer4 = self.conv_layer(self.f_dim, 1, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)

        # self.layer5 = self.conv_layer(self.f_dim, 1, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(height * width, 1),
            torch.nn.Sigmoid()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            # torch.nn.LeakyReLU(self.leak_value, inplace=True)
            torch.nn.ReLU(True)
        )
        return layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.attn1(out)
        out = self.layer3(out)
        # out = self.attn2(out)
        out = self.layer4(out)
        out = out.reshape(x.size(0),-1)
        out = self.layer5(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels, height, width):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, self.channels, self.height * self.width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.height, self.width).unsqueeze(2)

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
