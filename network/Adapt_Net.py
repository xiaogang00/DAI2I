import torch
import torch.nn as nn

class domain_invariant(nn.Module):
    def __init__(self, n_layers=2, num_classes=8):
        super(domain_invariant, self).__init__()
        layers = []
        curr_dims = 8
        norm_layer = nn.BatchNorm2d

        layers += [nn.Conv2d(3, curr_dims, 4, padding=1, stride=2)]
        for i in range(n_layers - 1):
            layers += [nn.Conv2d(curr_dims, curr_dims * 2, 4, padding=1, stride=2)]
            layers += [norm_layer(curr_dims * 2)]
            layers += [nn.LeakyReLU(0, True)]
            curr_dims *= 2
        layers_class = [nn.Linear(curr_dims, 64)]
        layers_class += [nn.LeakyReLU(0, True)]
        layers_class += [nn.Linear(64, num_classes)]
        self.main = nn.Sequential(*layers)
        self.main_class = nn.Sequential(*layers_class)

    def forward(self, x):
        feature = self.main(x)
        feature = feature.view(feature.shape[0], -1)
        output = self.main_class(feature)
        return feature, output

#######################################################################
v_siz = 9
z_siz = 128 - v_siz
class conv_mean_pool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(conv_mean_pool, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.pooling(out)
        return out


class mean_pool_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(mean_pool_conv, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.pooling(out)
        out = self.conv(out)
        return out


class upsample_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)

    def forward(self, x):
        out = x
        out = self.upsample(out)
        out = self.conv(out)
        return out


class residualBlock_down(nn.Module):  # for discriminator, no batchnorm
    def __init__(self, inplanes, outplanes):
        super(residualBlock_down, self).__init__()
        self.conv_shortcut = mean_pool_conv(inplanes, outplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.conv2 = conv_mean_pool(outplanes, outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class residualBlock_up(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(residualBlock_up, self).__init__()
        self.conv_shortcut = upsample_conv(inplanes, outplanes)
        self.conv1 = upsample_conv(inplanes, outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class _G_xvz(nn.Module):
    def __init__(self):
        super(_G_xvz, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)  # 3*128*128 --> 64*128*128
        self.resBlock0 = residualBlock_down(64, 64)  # 64*128*128 --> 64*64*64
        self.resBlock1 = residualBlock_down(64, 128)
        self.resBlock2 = residualBlock_down(128, 256)
        self.resBlock3 = residualBlock_down(256, 512)
        self.resBlock4 = residualBlock_down(512, 512)
        self.fc_v = nn.Linear(512 * 4 * 4, v_siz)
        self.fc_z = nn.Linear(512 * 4 * 4, z_siz)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.conv(x)
        out = self.resBlock0(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = out.view(-1, 512 * 4 * 4)
        v = self.fc_v(out)
        v = self.softmax(v)
        z = self.fc_z(out)

        return v, z


class _G_vzx(nn.Module):
    def __init__(self):
        super(_G_vzx, self).__init__()
        self.fc = nn.Linear(v_siz + z_siz, 4 * 4 * 512)
        self.resBlock1 = residualBlock_up(512, 512)  # 4*4-->8*8
        self.resBlock2 = residualBlock_up(512, 256)  # 8*8-->16*16
        self.resBlock3 = residualBlock_up(256, 128)  # 16*16-->32*32
        self.resBlock4 = residualBlock_up(128, 64)  # 32*32-->64*64
        self.resBlock5 = residualBlock_up(64, 64)  # 64*64-->128*128
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, v, z):
        x = torch.cat((v, z), 1)
        out = self.fc(x)  # out: 512*4*4
        out = out.view(-1, 512, 4, 4)  # (-1, 512, 4, 4)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = self.resBlock5(out)
        out = self.bn(out)
        out = self.ReLU(out)
        out = self.conv(out)
        out = self.tanh(out)

        return out


class view_point_transfer(nn.Module):
    def __init__(self):
        super(view_point_transfer, self).__init__()
        self.G_xvz = _G_xvz()
        self.G_vzx = _G_vzx()
        self.G_xvz = torch.nn.DataParallel(self.G_xvz)
        self.G_vzx = torch.nn.DataParallel(self.G_vzx)

    def forward(self, x, point_vector):
        padding_vector = torch.zeros_like(point_vector)
        padding_vector = padding_vector[:, 0:1]
        point_vector = torch.cat([padding_vector, point_vector, padding_vector], dim=1)

        v_bar, z_bar = self.G_xvz(x)
        output = self.G_vzx(point_vector, z_bar)
        return output

