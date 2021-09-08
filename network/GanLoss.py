import torch
from torch import nn


class LSGANLoss_multiScale(nn.Module):
    def __init__(self):
        super(LSGANLoss_multiScale, self).__init__()

    def __call__(self, fake_list, real_list, stage, **kwargs):
        loss = 0
        if stage == 'd':
            for fake, real in zip(fake_list, real_list):
                fake_label = torch.ones(fake.size()).cuda()
                real_label = torch.zeros(real.size()).cuda()
                loss += (fake - fake_label).norm(p=2) + (real - real_label).norm(p=2)
        elif stage == 'g':
            for fake in fake_list:
                real_label = torch.zeros(fake.size()).cuda()
                loss += (fake-real_label).norm(p=2)
        return loss
