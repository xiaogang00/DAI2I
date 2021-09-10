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
