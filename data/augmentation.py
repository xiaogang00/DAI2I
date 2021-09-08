import numbers

import random
import numpy as np
import PIL
import cv2

from skimage.transform import resize, rotate, SimilarityTransform, warp
from skimage.util import pad
import torchvision

import warnings

from skimage import img_as_ubyte, img_as_float
from PIL import Image


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
            ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant') for img in clip
            ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(img) for img in clip]

        return clip


class RandomResize_v1(object):
    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='bilinear', **kwargs):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class RandomResize_v2(object):
    """
    compared with RandomResize, this version treats width and height independently.
    """

    def __init__(self, ratio_w=(3. / 4., 4. / 3.), ratio_h=(3. / 4., 4. / 3.), interpolation='bilinear', **kwargs):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor_h = random.uniform(self.ratio_h[0], self.ratio_h[1])
        scaling_factor_w = random.uniform(self.ratio_w[0], self.ratio_w[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor_w)
        new_h = int(im_h * scaling_factor_h)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class RandomResize(object):
    def __init__(self, **kwargs):
        if kwargs['version'] == 'v1':
            self.main = RandomResize_v1(**kwargs)
        elif kwargs['version'] == 'v2':
            self.main = RandomResize_v2(**kwargs)
        else:
            raise NotImplementedError
    def __call__(self, clip):
        return self.main(clip)
        

class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


class CenterCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' + 'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = (im_w - w) // 2
        y1 = (im_h - h) // 2
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(clip[0], np.ndarray):
                if clip[0].max() > 1:
                    normalize = True
                else:
                    normalize = False
                    
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)
                img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array,
                                                                                                        img_as_float]


                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    if normalize:
                        jittered_img /= 255.
                    for i, func in enumerate(img_transforms):
                        # print(i, func)
                        jittered_img = func(jittered_img)
                    if normalize:
                        jittered_img *= 255.
                    jittered_clip.append(jittered_img.astype('float32'))

            elif isinstance(clip[0], PIL.Image.Image):
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)

                # Apply to all images
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img)

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class SelectRandomFrames(object):
    def __init__(self, consequent=False, number_of_frames=2):
        self.consequent = consequent
        self.number_of_frames = number_of_frames

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images for selection
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: List of number_of_frames images
        """
        frame_count = len(clip)
        num_frames_to_select = self.number_of_frames
        if self.consequent:
            first_frame = np.random.choice(max(1, frame_count - num_frames_to_select + 1), size=1)[0]
            selected = clip[first_frame:(first_frame + num_frames_to_select)]
        else:
            selected_index = np.sort(np.random.choice(range(frame_count), replace=True, size=num_frames_to_select))
            selected = clip[selected_index]
        return selected


class SplitSourceDriving(object):
    def __call__(self, video):
        source = np.array(video[:1], dtype='float32')
        video = np.array(video[1:], dtype='float32')
        return {'video': video.transpose((3, 0, 1, 2)),
                'source': source.transpose((3, 0, 1, 2))}


class VideoToTensor(object):
    """Convert video array to Tensor."""

    def __call__(self, driving):
        driving = np.array(driving, dtype='float32')
        return {'video': driving.transpose((3, 0, 1, 2))}


class RandomTranslation(object):
    def __init__(self, translation):
        if len(translation) != 4:
            raise ValueError('If degrees is a sequence,' 'it must be of len 2.')
        self.translation = translation

    def __call__(self, clip):
        translation_x = random.uniform(self.translation[0], self.translation[1])
        translation_y = random.uniform(self.translation[2], self.translation[3])
        changed = []
        if isinstance(clip[0], np.ndarray):
            for img in clip:
                tform = SimilarityTransform(scale=1, rotation=0, translation=(translation_x, translation_y))
                changed_img = warp(img, tform, preserve_range=True)
                changed.append(changed_img)
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return changed


class RandomShift(object):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias
    
    def __call__(self, image):
        image = image[0]
        y = random.uniform(-self.bias[0], self.bias[0])
        x = random.uniform(-self.bias[1], self.bias[1])
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        shifted = np.expand_dims(shifted, axis=0)
        return shifted


class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []
        self.select = SelectRandomFrames()
        self.transforms.append(self.select)

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))

        self.transforms.append(SplitSourceDriving())

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class AllAugmentationTransform_image:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, trans_param=None, shift_param=None, color_param=None):
        self.transforms = []
        if shift_param is not None:
            self.transforms.append(RandomShift(**shift_param))
        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))
        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))
        if trans_param is not None:
            self.transforms.append(RandomTranslation(**trans_param))
        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))
        if color_param is not None:
            self.transforms.append(ColorJitter(**color_param))
        if crop_param is not None:
            self.transforms.append(CenterCrop(**crop_param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        clip = np.array(clip)
        return clip
