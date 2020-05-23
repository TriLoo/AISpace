# -*- coding: utf-8 -*-

import mxnet as mx
import gluoncv as gcv
from gluoncv.model_zoo import get_model


def export_mod(model_name, model_prefix, data_shape, cls_num=1):
    net = get_model(model_name, pretrained_base=True)
    data = mx.nd.uniform(0.0, 1.0, (1, 3, data_shape, data_shape))
    net.hybridize()
    net(data)

    net.export()




if __name__ == '__main__':
    model_name = 'yolo3_mobilenet1.0_voc'
    pass
