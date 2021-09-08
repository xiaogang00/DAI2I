import argparse
import os
from data import attributeDataset, base_dataset
from util import util, opt, visualization
from collections import OrderedDict
import yaml


class dataset_now(base_dataset.BaseDataset):

    def __init__(self, src_dataset, dst_dataset):
        super(dataset_now, self).__init__()
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset

    def __getitem__(self, index):
        src_img, attr = self.src_dataset[index % len(self.src_dataset)]
        dst_img = self.dst_dataset[index % len(self.dst_dataset)]
        return src_img, dst_img, attr

    def __len__(self):
        return max(len(self.src_dataset), len(self.dst_dataset))


class dataset_now2(base_dataset.BaseDataset):

    def __init__(self, src_dataset, dst_dataset):
        super(dataset_now2, self).__init__()
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset

    def __getitem__(self, index):
        src_img, attr = self.src_dataset[index % len(self.src_dataset)]
        dst_img = self.dst_dataset[index % len(self.dst_dataset)]
        return src_img, dst_img, attr

    def __len__(self):
        return min(len(self.src_dataset), len(self.dst_dataset))


class Engine(object):

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-cp', '--config_path', default=None)
        parser.add_argument('-gpu', default='0', help='the gpu index to run')
        parser.add_argument('-sp', '--save_dir', default='checkpoints/default2', help='model save path')
        parser.add_argument('-ct', '--continue_train', action='store_true',
                            help='if true, then load model stored in the save_dir')
        return parser

    def basic_setting(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        util.mkdir(self.args.save_dir)

    def define_model(self):
        from network import stargan_model
        from basemodel.stargan import get_rafd_stargan

        ## stargan
        base_model = get_rafd_stargan(version=self.config['basemodel_version'])

        ## generator
        from network import adain_net
        from torch import nn
        G = adain_net.Generator_split_v2(**self.config['generator'])
        G = nn.DataParallel(G)

        ## discriminator
        D = stargan_model.Discriminator_split_sn_multiScale(**self.config['discrim'])

        ## classification network
        from network import Adapt_Net
        class_num = len(self.config['attr_names'])
        classifier = Adapt_Net.domain_invariant(7, num_classes=class_num)
        classifier = nn.DataParallel(classifier)

        return base_model, G, D, classifier

    def define_optim(self, model):
        from optimizer import optim_trainable_space
        train_config = opt.opt()
        train_config.merge_dict(self.config)
        print(train_config)

        optim = optim_trainable_space.optimizer(model, train_config)

        return optim

    def load_dataset(self):
        with open(self.config['src_train_list'], 'r') as f:
            train_list = [os.path.join(self.config['src_data_dir'], tmp.rstrip()) for tmp in f]
        train_dataset = attributeDataset.GrouppedAttrDataset(image_list=train_list, attributes=self.config['attr'],
                                                             csv_path=self.config['csv_path'], aug=self.config['src_aug'],
                                                             crop_size=self.config['src_crop_size'], aug_param=self.config['aug_param'])


        from data import datasetSimple
        import glob
        dst_list = glob.glob(self.config['dst_dir'] + '/*.jpg')
        dst_dataset = datasetSimple.Dataset(image_list=dst_list,
                                            crop_size=(self.config['dst_crop_size'], self.config['dst_crop_size']),
                                            aug=self.config['dst_aug'], aug_param=self.config['dst_aug_param'])


        join_train_dataset = dataset_now(train_dataset, dst_dataset)
        fid_dataset = dataset_now2(train_dataset, dst_dataset)
        return join_train_dataset, fid_dataset
        
    def train(self):
        model = self.define_model()
        optim = self.define_optim(model)
        train_dataset, fid_dataset = self.load_dataset()
        optim.fid_dataset = fid_dataset
        from util import training_framework
        if 'framework_config' in self.config:
            framework_config = self.config['framework_config']
        else:
            framework_config = {}

        TrainEngine = training_framework.TrainEngine(dataset=train_dataset, optimizer=optim,
                                                     batch_size=self.config['batch_size'], 
                                                     config=framework_config)
        TrainEngine.run(self.config['epoch'])

    def _default_config(self, config):
        return config

    def run(self):
        parser = self.parse_args()
        self.args = parser.parse_args()
        self.config_this = OrderedDict()
        if self.args.config_path is None:
            self.args.config_path = '{}/options.yaml'.format(self.args.save_dir)
        with open(self.args.config_path, 'r') as f:
            self.config = yaml.load(f)
        self.config = self._default_config(self.config)
        self.config['save_dir'] = self.args.save_dir
        self.config['continue_train'] = self.args.continue_train
        
        self.basic_setting()
        self.train()

if __name__ == '__main__':
    engine = Engine()
    engine.run()
