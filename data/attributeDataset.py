from . import base_dataset
import scipy.misc
import numpy as np
import torch
import pandas as pd
import os
import torchvision as tv
from util import util
from util.logger import logger
import data.augmentation as augmentation


log = logger()
mean = torch.Tensor((0.5, 0.5, 0.5))
stdv = torch.Tensor((0.5, 0.5, 0.5))
forward_transform = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])


class GrouppedAttrDataset(base_dataset.BaseDataset):
    def __init__(self, image_list, attributes, transform=forward_transform, scale=(128, 128), crop_size=(160, 160),
                 bias=(0, 0), aug=False, aug_param={},
                 csv_path='', csv_split=','):
        super(GrouppedAttrDataset, self).__init__()
        self.files = []
        supported_format = ['jpg', 'png', 'jpeg']
        for image_now in image_list:  # filter out files that are not image
            format = image_now.split('.')[-1]
            format = format.lower()
            is_image = False
            for sf in supported_format:
                if format == sf:
                    is_image = True
                    break
            if is_image:
                self.files += [image_now]

        print('* Total Images: {}'.format(len(self.files)))
        self.transform = transform
        self.scale = scale
        self.bias = bias
        self.frame = pd.read_csv(csv_path, sep=csv_split)
        self.crop_size = crop_size

        attributes = attributes.split(',')
        f2 = self.frame['name'].to_frame()
        f3 = self.frame.replace(-1, 0)
        for attrs in attributes:
            attrs_split = attrs.split('@')
            for i, att in enumerate(attrs_split):
                if att[0] == '#':
                    col_now = ~f3[att[1:]]
                else:
                    col_now = f3[att]
                if i == 0:
                    attr_value = pd.DataFrame(col_now)
                else:
                    attr_value = pd.concat([attr_value, col_now], axis=1)
            attr_value = list(attr_value.values.astype(np.float32))
            f2[attrs] = attr_value
        self.frame = f2

        self.aug = aug
        flip_param = {}
        flip_param['horizontal_flip'] = True
        flip_param['time_flip'] = False
        if 'resize_param' not in aug_param:
            resize_param = {}
            resize_param['ratio'] = [0.8, 1.2]
            aug_param['resize_param'] = resize_param
        if 'rotation_param' not in aug_param:
            rotation_param = {'degrees': [-10, 10]}
            aug_param['rotation_param'] = rotation_param
        if 'shift_param' not in aug_param:
            shift_param = {'bias':[50,50]}
            aug_param['shift_param'] = shift_param
        if 'crop_param' not in aug_param:
            aug_param['crop_param'] = None

        self.augmentation = augmentation.AllAugmentationTransform_image(resize_param=aug_param['resize_param'],
                                                                        rotation_param=aug_param['rotation_param'],
                                                                        flip_param=flip_param,
                                                                        crop_param=aug_param['crop_param'],
                                                                        trans_param=None, 
                                                                        shift_param=aug_param['shift_param'])

    def __getitem__(self, index):
        try:
            img = util.readRGB(self.files[index]).astype(np.float32)
            if self.aug:
                img = np.expand_dims(img, axis=0)
                img = self.augmentation(img)
                img = np.squeeze(img, axis=0)
                if self.scale[0] > 0:
                    img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
                img = self.transform(img.astype(np.uint8))
            else:
                if self.crop_size[0] > 0:
                    img = util.center_crop(img, self.crop_size, self.bias)
                if self.scale[0] > 0:
                    img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
                img = self.transform(img)
            image_name = os.path.basename(self.files[index])
            attribute = self.frame[self.frame['name'] == image_name]
            attribute = attribute.values[0][1:]
            attribute = tuple(attribute)
            return img, attribute

        except:
            rd_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rd_idx)

    def __len__(self):
        return len(self.files)

