'''
this script contains the optimizer class.
'''

from __future__ import print_function
import torch
from torch import nn
from collections import OrderedDict
import torch.optim as optim
from util.opt import opt
import os


class base_optimizer(nn.Module):
    def __init__(self):
        super(base_optimizer, self).__init__()

    def _check_and_load(self, net, path):
        if os.path.exists(path):
            print('loading {}'.format(path))
            net.load_state_dict(torch.load(path))
        else:
            print('warning: cannot load ' + path)

    def get_current_errors(self):
        return self.loss

    def add_summary(self, global_step):
        d = self.get_current_errors()
        for keys, values in zip(d.keys(), d.values()):
            values = float(values.data)
            self.writer.add_scalar(keys, values, global_step=global_step)

    def add_summary_heavy(self, epoch):
        pass

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

    def print_network(self, net, name):
        with open(self.opt.save_dir + '/{}.txt'.format(name), 'w') as f:
            print(net, file=f)
