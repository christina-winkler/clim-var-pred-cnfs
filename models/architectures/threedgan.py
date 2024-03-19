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

        self.unet3d = unet3d.UNet3D(in_channel=1)

        padd = (1, 1, 1)
        # note +1 as we only concatenate one noise layer
        self.layer1 = self.conv_layer(3, 2, kernel_size=3, stride=1, padding=padd, bias=self.bias)
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
        out1 = self.layer1(out)#.permute(0,2,1,3,4)
        out2 = self.unet3d(out)
        import pdb; pdb.set_trace()

        # out = self.layer2(out)
        # out = self.attn1(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.attn2(out)
        # out = self.layer5(out)
        # import pdb; pdb.set_trace()
        return out


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
