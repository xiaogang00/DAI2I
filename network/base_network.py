from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from util import util
from torch.utils import model_zoo
from util import opt
from util.logger import logger

log = logger(True)


class BaseModel(object):

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.continue_train = False
        self.opt.lr = 1e-3

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self, x):
        pass

    def add_summary(self, global_step):
        d = self.get_current_errors()
        for key, value in d.items():
            self.writer.add_scalar(key, value, global_step)

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def print_current_errors(self, epoch, i, record_file=None, print_msg=True):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        if print_msg:
            print(message)
        if record_file is not None:
            with open(record_file + '/loss.txt', 'w') as f:
                print(message, file=f)
        return message

    def save(self, label):
        pass

    def load(self, pretrain_path, label):
        pass

    def save_network(self, network, save_dir, save_name):
        save_path = os.path.join(save_dir, save_name)
        print('saving %s in %s' % (save_name, save_dir))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    def resume_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        print('loading %s from %s' % (save_filename, save_path))
        network.load_state_dict(torch.load(save_path))

    def load_network(self, pretrain_path, network, file_name):
        save_path = os.path.join(pretrain_path, file_name)
        print('loading %s from %s' % (file_name, save_path))
        network.load_state_dict(torch.load(save_path))

    def print_network(self, net, filepath=None):
        if filepath is None:
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('Total number of parameters: %d' % num_params)
        else:
            num_params = 0
            with open(filepath + '/network.txt', 'w') as f:
                for param in net.parameters():
                    num_params += param.numel()
                print(net, file=f)
                f.write('Total number of parameters: %d' % num_params)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class VGG(nn.Module, BaseModel):
    def __init__(self, pretrained=True, local_model_path=None, nChannel=64):
        super(VGG, self).__init__()
        self.features_1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, nChannel, kernel_size=3, padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2_1', nn.Conv2d(nChannel, nChannel * 2, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(nChannel * 2, nChannel * 2, kernel_size=3, padding=1)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3_1', nn.Conv2d(nChannel * 2, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
        ]))
        self.features_2 = nn.Sequential(OrderedDict([
            ('conv3_2', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(nChannel * 4, nChannel * 4, kernel_size=3, padding=1)),
            ('relu3_5', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),
            ('conv4_1', nn.Conv2d(nChannel * 4, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
        ]))
        self.features_3 = nn.Sequential(OrderedDict([
            ('conv4_2', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('conv4_4', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu4_4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),
            ('conv5_1', nn.Conv2d(nChannel * 8, nChannel * 8, kernel_size=3, padding=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
        ]))
        if pretrained:
            if local_model_path is None:
                print('loading default VGG')
                model_path = 'https://www.dropbox.com/s/4lbt58k10o84l5h/vgg19g-4aff041b.pth?dl=1'
                state_dict = torch.utils.model_zoo.load_url(model_path, 'checkpoints/vgg')
            else:
                print('loading VGG from %s' % local_model_path)
                state_dict = torch.load(local_model_path)
            model_dict = self.state_dict()
            state_dict = {key: value for key, value in state_dict.items() if key in model_dict}
            # print(state_dict.keys())
            self.load_state_dict(state_dict)

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3

