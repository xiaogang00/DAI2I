import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as sn
from network import self_attention

def concat(x, c):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.expand(-1, -1, x.size(2), x.size(3))
    return torch.cat((x, c), dim=1)

class none_layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class ResidualBlock_sn(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock_sn, self).__init__()
        self.main = nn.Sequential(
            sn(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, in_channels=3, out_channels=3, n_down=2,
                 condition=False, act_type='relu', n_SA_layer=0):
        super(Generator, self).__init__()
        if condition:
            in_channels += 2

        if act_type == 'relu':
            act_fn = nn.ReLU
        elif act_type == 'lrelu':
            act_fn = nn.LeakyReLU

        layers = []
        layers.append(nn.Conv2d(in_channels + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(act_fn(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(n_down):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(act_fn(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(n_SA_layer):
            layers.append(self_attention.NONLocalBlock2D(curr_dim))

        # Up-sampling layers.
        for i in range(n_down):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(act_fn(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        self.condition = condition

    def forward(self, x, domain_cond=None):
        if self.condition:
            x = concat(x, domain_cond)
        # Replicate spatially and concatenate domain information.
        return self.main(x)


class sn_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm):
        super().__init__()
        if norm == 'none':
            norm_fn = none_layer
        elif norm == 'in':
            norm_fn = nn.InstanceNorm2d
        self.main = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            norm_fn(out_channels), 
            nn.LeakyReLU(0.01)
        )
    def forward(self, x):
        return self.main(x)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm):
        super().__init__()
        if norm == 'none':
            norm_fn = none_layer
        elif norm == 'in':
            norm_fn = nn.InstanceNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_fn(out_channels), 
            nn.LeakyReLU(0.01)
        )
    def forward(self, x):
        return self.main(x)


class Discriminator_sn_plain_multiScale(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6, in_channels=3, norm='none', condition=False, n_scale=3):
        super(Discriminator_sn_plain_multiScale, self).__init__()
        layers = []
        if condition:
            in_channels += 2

        layers.append(sn_conv_block(in_channels, conv_dim, kernel_size=4, stride=2, padding=1, norm=norm))
        curr_dim = conv_dim
        out_layers = []
        for i in range(1, repeat_num):
            layers.append(sn_conv_block(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, norm=norm))
            curr_dim = curr_dim * 2
            if i >= repeat_num-n_scale:
                out_layers.append(sn(nn.Conv2d(curr_dim, 1, kernel_size=1)))
                

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.ModuleList(layers)
        self.out_layers = nn.ModuleList(out_layers)
        self.conv1 = sn(nn.Conv2d(curr_dim, 1, kernel_size=kernel_size))
        self.condition = condition
        self.repeat_num = repeat_num
        self.n_scale = n_scale

    def forward(self, x, cond=None, *args):
        if self.condition:
            x = concat(x, cond)
        out = []
        j = 0
        for i in range(self.repeat_num):
            x = self.main[i](x)
            if i >= self.repeat_num-self.n_scale:
                out.append(self.out_layers[j](x))
                j += 1
        out_final = self.conv1(x)
        out.append(out_final)
        return out


class Discriminator_split_sn_multiScale(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=6, in_channels=3, norm='none', n_scale=3):
        super(Discriminator_split_sn_multiScale, self).__init__()
        self.locals = locals().copy()
        self.main1 = Discriminator_sn_plain_multiScale(image_size=image_size, conv_dim=conv_dim, repeat_num=repeat_num, in_channels=in_channels, norm=norm, n_scale=n_scale)
        self.main2 = Discriminator_sn_plain_multiScale(image_size=image_size, conv_dim=conv_dim, repeat_num=repeat_num, in_channels=in_channels, norm=norm, n_scale=n_scale)
    
    def forward(self, x, cond):
        if cond == 0:
            return self.main1(x)
        elif cond == 1:
            return self.main2(x)
        else:
            raise NotImplementedError

