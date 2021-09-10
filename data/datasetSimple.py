from data import base_dataset
from data.attributeDataset import forward_transform,  util, scipy, os, np
import glob
import data.augmentation as augmentation


class Dataset(base_dataset.BaseDataset):
    def __init__(self, image_list, transform=forward_transform, scale=(128, 128), crop_size=(0, 0),
                 bias=(0, 0), aug=False, return_filename=False, aug_param={}):
        super(Dataset, self).__init__()
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
        self.crop_size = crop_size
        self.return_filename = return_filename
        self.aug = aug
        if aug:
            flip_param = {}
            flip_param['horizontal_flip'] = True
            flip_param['time_flip'] = False
            if 'resize_param' not in aug_param:
                print('resize param not in aug param')
                resize_param = {}
                resize_param['ratio_h'] = [0.8, 1.2]
                resize_param['ratio_w'] = [0.8, 1.2]
                resize_param['ratio'] = [0.8, 1.2]
                resize_param['version'] = 'v1'
                aug_param['resize_param'] = resize_param
            if 'rotation_param' not in aug_param:
                rotation_param = {'degrees': [-10, 10]}
                aug_param['rotation_param'] = rotation_param
            if 'shift_param' not in aug_param:
                shift_param = {'bias': [50, 50]}
                # shift_param = None
                aug_param['shift_param'] = shift_param
            if 'crop_param' not in aug_param:
                aug_param['crop_param'] = None
            if 'color_param' not in aug_param:
                aug_param['color_param'] = None
            self.augmentation = augmentation.AllAugmentationTransform_image(resize_param=aug_param['resize_param'],
                                                                            rotation_param=aug_param['rotation_param'],
                                                                            flip_param=flip_param,
                                                                            crop_param=aug_param['crop_param'],
                                                                            trans_param=None,
                                                                            shift_param=aug_param['shift_param'],
                                                                            color_param=aug_param['color_param'])

    def __getitem__(self, index):
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
        if self.return_filename:
            return img, os.path.basename(self.files[index])
        else:
            return img

    def __len__(self):
        return len(self.files)


