import torch
from torch import nn
from network import self_attention
from network.cls_net import classifier as im2style_net
from network import stargan_model

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Conv2d(style_dim, in_channel * 2, kernel_size=1, padding=0)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.innorm = AdaptiveInstanceNorm(out_channels, style_dim)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, style):
        x = self.conv(x)
        x = self.innorm(x, style)
        x = self.lrelu(x)
        return x


class DeonvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride=stride, padding=padding)
        self.innorm = AdaptiveInstanceNorm(out_channels, style_dim)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, style):
        x = self.deconv(x)
        x = self.innorm(x, style)
        x = self.lrelu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = AdaptiveInstanceNorm(dim_out, style_dim)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = AdaptiveInstanceNorm(dim_out, style_dim)

    def forward(self, x, style):
        y = self.conv1(x)
        y = self.in1(y, style)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.in2(y, style)
        y = y + x
        return y


class Generator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, in_channels=3, out_channels=3, n_down=2, style_dim=256, n_SA_layer=0):
        super().__init__()
        layers = []
        layers.append(ConvBlock(3, conv_dim, 7, 1, 3, style_dim))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(n_down):
            layers.append(ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, style_dim))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim,
                                        dim_out=curr_dim, style_dim=style_dim))
        for i in range(n_SA_layer):
            layers.append(self_attention.NONLocalBlock2D(curr_dim))
        # Up-sampling layers.
        for i in range(n_down):
            layers.append(DeonvBlock(curr_dim, curr_dim//2, 4, 2, 1, style_dim))
            curr_dim = curr_dim // 2

        self.main = nn.ModuleList(layers)
        self.out = nn.Sequential(
            nn.Conv2d(curr_dim, out_channels, kernel_size=7,stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, style):
        y = x
        for i in range(len(self.main)):
            y = self.main[i](y, style)
        y = self.out(y)
        return y


class Generator_split(nn.Module):

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, in_channels=3, out_channels=3, n_down=2, condition=False, act_type='lrelu', n_SA_layer=0):
        self.locals = locals().copy()
        super(Generator_split, self).__init__()
        style_dim = 1024
        self.main1 = stargan_model.Generator(conv_dim, c_dim, repeat_num, in_channels, out_channels,n_down=n_down, condition=False, act_type=act_type, n_SA_layer=n_SA_layer)
        self.main2 = Generator(conv_dim=conv_dim, repeat_num=repeat_num, in_channels=in_channels, out_channels=out_channels, n_down=n_down, style_dim=style_dim, n_SA_layer=n_SA_layer)
        self.im2style = im2style_net(n_layer=4, channels=64, num_classes=style_dim)

    def forward(self, x, cond=False, style_img=None):
        if cond == 0:
            return self.main1(x)
        elif cond == 1:
            _, style = self.im2style(style_img)
            style = style.unsqueeze(2).unsqueeze(3)
            return self.main2(x, style)
        else:
            raise NotImplementedError
        

class Generator_split_v2(nn.Module):

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, in_channels=3, out_channels=3, n_down=2, condition=False, act_type='lrelu', n_SA_layer=0):
        self.locals = locals().copy()
        super(Generator_split_v2, self).__init__()
        style_dim = 256
        self.main1 = Generator(conv_dim=conv_dim, repeat_num=repeat_num, in_channels=in_channels, out_channels=out_channels, n_down=n_down, style_dim=style_dim, n_SA_layer=n_SA_layer)
        self.main2 = Generator(conv_dim=conv_dim, repeat_num=repeat_num, in_channels=in_channels, out_channels=out_channels, n_down=n_down, style_dim=style_dim, n_SA_layer=n_SA_layer)
        self.im2style = im2style_net(n_layer=4, channels=16, num_classes=style_dim)

    def forward(self, x, cond=False, style_img=None):
        if cond == 0:
            _, style = self.im2style(style_img)
            style = style.unsqueeze(2).unsqueeze(3)
            return self.main1(x, style)
        elif cond == 1:
            _, style = self.im2style(style_img)
            style = style.unsqueeze(2).unsqueeze(3)
            return self.main2(x, style)
        else:
            raise NotImplementedError
