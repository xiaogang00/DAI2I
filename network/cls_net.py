import torch
from torch import nn

class conv_block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(*args, **kwargs),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.main(x)

class classifier(nn.Module):
    def __init__(self, n_layer, channels=64, num_classes=8):
        super().__init__()
        self.main = []
        self.main.append(conv_block(3, channels, 7, 1, padding=3))
        channels_now = channels
        for i in range(n_layer):
            self.main.append(conv_block(channels_now, channels_now * 2, kernel_size=4, stride=2, padding=1))
            channels_now *= 2
        self.main.append(nn.AdaptiveAvgPool2d((1,1)))
        self.main = nn.Sequential(*self.main)
        self.cls = nn.Conv2d(channels_now, num_classes, 1, 1, 0)
    def forward(self, x):
        feat = self.main(x)
        y = self.cls(feat).squeeze(3).squeeze(2)
        return feat, y
