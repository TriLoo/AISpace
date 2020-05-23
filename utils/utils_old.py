# -*- coding: utf-8 -*-

## @author : smh
## @date   : 2019.04.17
## @brief  : Training the finetuned model

import mxnet as mx
from mxnet import autograd, gluon
from mxnet.gluon.data import dataset
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import experimental
import gluoncv as gcv

import os
import numpy as np
import xml.etree.ElementTree as ET
import logging
import warnings


# check file existance
def check_file(file_name):
    if not os.path.exists(file_name):
        return False
    else:
        return True


class SSDDefaultTrainTransform_v2(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        from gluoncv.model_zoo.ssd.target import SSDTargetGenerator_v2
        self._target_generator = SSDTargetGenerator_v2(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _, gt_boxes_parsed = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0], gt_boxes_parsed[0]


class VisionDataset(dataset.Dataset):
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir. Did you forget to initialize \
                          datasets described in: \
                          `http://gluon-cv.mxnet.io/build/examples_datasets/index.html`? \
                          You need to initialize each dataset only once.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)


# get the dataset
class VOCDetection(VisionDataset):
    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    # CLASSES = ('general', 'vehicle', 'animal', 'person', 'logo')
    # CLASSES = ('person', 'face', 'vehicle', 'animal', 'general', 'text', 'logo')
    # CLASSES = ('general', 'vehicle', 'animal', 'person')
    # CLASSES = ('person', 'face', 'vehicle', 'animal', 'general', 'logo')
    CLASSES = ('guohui',)

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        # print('current img: ', img_path)
        # print('shape of current img: ', img.shape)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))

        return [self._load_label(idx) for idx in range(len(self))]


# dataloader for training
def get_traindataloader(net, train_dataset, data_shape, batch_size, num_workers, is_shuffle=True):
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    #print(anchors)  # mxnet ndarray, shape: 1 * 6132 * 4
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, is_shuffle, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    return train_loader


# used custrom SSDDefaultTrainTransform
def get_traindataloader_v2(net, train_dataset, data_shape, batch_size, num_workers, is_shuffle=True):
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors, _ = net(mx.nd.zeros((1, 3, height, width)))
    #print(anchors)  # mxnet ndarray, shape: 1 * 6132 * 4
    # batchify_fn = Tuple(Stack(), Stack(), Stack(), Pad(pad_val=-1))  # stack image, cls_targets, box_targets
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets, box_gt
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform_v2(width, height, anchors)),
        batch_size, is_shuffle, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    return train_loader


# dataloader for evaluation
def get_evaldataloader(val_dataset, data_shape, batch_size, num_workers):
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=num_workers)

    return val_loader


class SSDOriTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    # def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    def __init__(self, width, height, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(img.dtype)


def get_ori_evaldataloader(val_dataset, data_shape, batch_size, num_workers):
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDOriTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=num_workers)

    return val_loader


if __name__ == '__main__':
    dataset_dir = '../datasets/cocovoc_cartoon/'
    test_set = VOCDetection(root=dataset_dir, splits=[(2019, 'train')])
    classes = [str(i) for i in range(6)]
    model_name = 'ssd_320_mnasnet_1a_dilated'
    net = gcv.model_zoo.get_model(model_name, pretrained=False, classes=classes)
    net.initialize()

    # test_dataloader = get_evaldataloader(test_set, 320, 8, num_workers=2)
    # test_dataloader = get_traindataloader(net, test_set, 320, 8, 2)
    test_dataloader = get_traindataloader_v2(net, test_set, 320, 8, 2)
    ctx = [mx.cpu(31), mx.cpu(32)]

    for databatch in test_dataloader:
        print('type of databatch: ', type(databatch))
        print('len of databatch: ', len(databatch))
        print('shape of data: ', databatch[0].shape)       # (8, 3, 320, 320)
        print('shape of label: ', databatch[1].shape)      # (8, N, 6) N is the maximum objs nums

        # ----------- for eval dataset -------------
        # data_list = gluon.utils.split_and_load(databatch[0], ctx)
        # label_list = gluon.utils.split_and_load(databatch[1], ctx)
        # box_list = gluon.utils.split_and_load(databatch[2], ctx)
        # print('len of data_list: ', len(data_list))                     # 2
        # print('shape of data_list element: ', data_list[0].shape)       # (4, 3, 320, 320)
        # print('shape of label_list element: ', label_list[0].shape)       # (4, N, 6)
        # print('shape of bboxes_list element: ', box_list[0].shape)       # (4, N, 6)

        # ----------- for train dataset -------------
        data_list = gluon.utils.split_and_load(databatch[0], ctx)
        label_list = gluon.utils.split_and_load(databatch[1], ctx)
        box_list = gluon.utils.split_and_load(databatch[2], ctx)
        print('len of data_list: ', len(data_list))                      # 2
        print('shape of data_list element: ', data_list[0].shape)        # (4, 3, 320, 320)
        print('shape of label_list element: ', label_list[0].shape)      # (4, N), N is the # of objs in the images
        print('shape of bboxes_list element: ', box_list[0].shape)       # (4, N, 4), N is the # of objs in the images

        break
