import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from models.architectures import conv_lstm

"""
Implements model class of our 3D GAN model with conditional generative prior. 
"""

class ConvLayer3D(nn.Conv3d):
    def __init__(self, input_dim, output_dim,  kernel_size=3, stride=2, padding=(0,0,0), bias=False):
        super().__init__(input_dim, output_dim,  kernel_size, stride, padding, bias)    

        self.layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, 
                                     stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            # torch.nn.ReLU(True)
            torch.nn.LeakyReLU(0.4, True)
        )
    
    def forward(self, x):
        return self.layer(x)


class GaussianPrior(nn.Module):
    def __init__(self, in_c, cond_channels, final=False):
        super(GaussianPrior, self).__init__()
        self.cond_channels = cond_channels
        self.conv = ConvLayer3D(input_dim=1, output_dim=2, stride=1, padding=(1,1,1))

    def final_prior(self, feat_map):
        h = self.conv(feat_map)
        mean, sigma = h[:, 0].unsqueeze(1), nn.functional.softplus(h[:, 1].unsqueeze(1).type(torch.DoubleTensor).cuda())
        return mean, sigma

    def forward(self, feat_map, eps=1.0):
        # sample from conditional prior
        mean, sigma = self.final_prior(feat_map)
        prior = torch.distributions.normal.Normal(loc=mean, scale=sigma*eps+0.00001)
        z = prior.sample().type(torch.FloatTensor).cuda()
        return z

class Generator(torch.nn.Module):
    def __init__(self, in_c, out_c, height, width):
        super(Generator, self).__init__()
        self.bias =  False # params.bias
        self.f_dim = 64
        self.init = True
        self.cond_prior = GaussianPrior(in_c=1, cond_channels=64)
        self.conv_lstm = conv_lstm.ConvLSTMCell(in_channels=2, hidden_channels=32, out_channels=4*1, num_layers=3).to('cuda')

        padd = (1, 1, 1)
        self.deconv = torch.nn.ConvTranspose3d(2, self.f_dim, 3)
        self.layer1 = ConvLayer3D(2, self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        self.layer2 = ConvLayer3D(self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        self.attn1 = SelfAttention(2*self.f_dim, height, width)
        self.layer3 = ConvLayer3D(2*self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        self.layer4 = ConvLayer3D(2*self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        self.attn2 = SelfAttention(self.f_dim, height, width)
        self.layer5 = ConvLayer3D(self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=padd, bias=self.bias)
        # self.layer6 = ConvLayer3D(2*self.f_dim, 1, kernel_size=3, stride=1, padding=padd, bias=self.bias)

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )
        # block8 = [UpsampleBlock(64, 1) for _ in range(2)]
        # block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        # self.block8 = nn.Sequential(*block8)

    def forward(self, x, state=None):

        # process context window
        if self.init or state==None:
            state = (torch.zeros_like(x), torch.zeros_like(x))
            (h,c) = self.conv_lstm(x, state)
        else:
            (h,c) = self.conv_lstm(x, state)
        
        # sample z from base density
        z = self.cond_prior(h)

        # pass z through the generator networks
        out1 = self.layer1(z.permute(0,2,1,3,4).contiguous())
        # print(out1.shape)
        out2 = self.layer2(out1)
        # print(out2.shape)
        out3 = self.attn1(out2)
        # print(out3.shape)
        out4 = self.layer3(out2)
        # print(out4.shape)
        out5 = self.layer4(out4)
        # print(out5.shape)
        out6 = self.attn2(out5)
        # print(out6.shape)
        out7 = self.layer5(out5)
        # pdb.set_trace()
        # out8 = self.layer6(out7)
        block1 = self.block1(out7.squeeze(2))
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block1 + block6)
        # block8 = self.block8(block1 + block7)
        return block7.unsqueeze(1) / 2 #(torch.tanh(block8) + 1) / 2


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
        self.attn1 = SelfAttention(2*self.f_dim, height, width)
        self.layer3 = self.conv_layer(2*self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=(1,1,1), bias=self.bias)
        self.attn2 = SelfAttention(self.f_dim, height, width)
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
        out = self.attn1(out)
        out = self.layer3(out)
        out = self.attn2(out)
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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x