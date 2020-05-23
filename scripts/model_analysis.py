# -*- coding: utf-8 -*-

import mxnet as mx
import gluoncv as gcv
from mxnet import autograd, nd
import numpy as np

import common
classes = common.mapped_cls

def show_single_img(img_file, model_name, param_file, do_train=False):
    net = gcv.model_zoo.get_model(model_name, pretrained=False, classes=classes)
    net.load_parameters(param_file)
    data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=320)

    if not do_train:
        eval_output = net(data)
        cls_preds, bbox_preds, _ = eval_output
    else:
        with autograd.record():
            train_output = net(data)

        cls_preds, bbox_preds, _ = train_output

    print('shape of cls_preds: ', cls_preds.shape)
    print('bbox_preds shape: ', bbox_preds.shape)


def viz_net(model_name, data_shape, viz_prefix):
    net = gcv.model_zoo.get_model(model_name, pretrained=False, classes=classes)
    net.initialize()
    # print(net)
    gcv.utils.viz.plot_network(net, shape=(1, 3, data_shape, data_shape), save_prefix=viz_prefix)

    # data = mx.sym.Variable('data')
    # net = net(data)
    # net = mx.sym.Group(list(net))
    # mx.viz.plot_network(net, title=viz_prefix, shape={'data':(1, 3, data_shape, data_shape)})


if __name__ == '__main__':
    # ----------- ************* -------------
    # Show single image using gcv net
    # ----------- ************* -------------
    img_file = './badcases/all_badcases/df92-iieqapu5371410.jpg'
    model_name = 'ssd_320_mnasnet_1a_dilated'
    param_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon/mnasnet1a_ssd_320_dilated_cartoon_v7_temp_mixup_best.params'
    # show_single_img(img_file, model_name, param_file)

    # ----------- ************* -------------
    #  visualize the model
    # ----------- ************* -------------
    model_name = 'efficientdet_d0'
    # model_name = 'efficientnet'
    data_shape = 512
    viz_prefix = model_name

    viz_net(model_name, data_shape, viz_prefix)

