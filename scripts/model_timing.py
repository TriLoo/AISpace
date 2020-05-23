# -*- coding: utf-8 -*-

import time
import mxnet as mx
import gluoncv as gcv
import os

import common
# classes = common.mapped_cls
# classes = ['bdbk']
classes = ['guohui']


if __name__ == '__main__':
    ctx = mx.cpu(39)
    # data = mx.nd.random.uniform(0.0, 1.0, (1, 3, 320, 320))
    data = mx.nd.random.uniform(0.0, 1.0, (1, 3, 512, 512))
    # data = mx.nd.random.uniform(0.0, 1.0, (1, 3, 300, 300))
    data = data.as_in_context(ctx)

    # model_name = 'ssd_300_mobilenet0.25_coco'

    # model_name = 'ssd_300_mnasnet_1a'
    # model_name = 'ssd_300_mnasnet_1a_normal'
    # model_name = 'ssd_320_mnasnet_1a_large_head'
    # model_name = 'ssd_320_mnasnet_1a_dilated'
    model_name = 'ssd_300_mnasnet_1a'

    net = gcv.model_zoo.get_model(model_name, pretrained=False, classes=classes)
    # net = gcv.model_zoo.get_model(model_name, pretrained=False)
    # net.initialize()
    # net.reset_class(classes=classes)

    # weight_file = './pretrained/mnasnet1a_ssd_320_dilated_mixup_schedule_best.params'
    # weight_file = './pretrained/ssd_mobilenet_reduced_0_75_cocovoc_300_using_pretrained_512_best.params'
    # weight_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon/mnasnet1a_ssd_320_dilated_cartoon_v3_mixup_best.params'
    # weight_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon/mnasnet1a_ssd_320_dilated_cartoon_v5_mixup_best.params'
    # weight_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon_test/mnasnet1a_ssd_320_dilated_cartoon_best.params'
    # weight_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon/mnasnet1a_ssd_320_dilated_cartoon_v8_best.params'
    # weight_file = './scripts/mnasnet1a_ssd_320_dilated_cartoon/mnasnet1a_ssd_320_dilated_cartoon_v9_mixup_0022_0.5487.params'

    # weight_file = './scripts/mnasnet1a_ssd_bdbk/mnasnet1a_ssd_512_best.params'
    weight_file = './scripts/mnasnet1a_ssd_guohui/mnasnet1a_ssd_512_0032_0.9089.params'

    net.load_parameters(weight_file)

    net.hybridize()
    net.collect_params().reset_ctx(ctx)

    # warm up
    for i in range(5):
        net(data)
        mx.nd.waitall()

    # timing
    print('Start timing ...')
    start_time = time.time()
    for i in range(100):
        net(data)
        mx.nd.waitall()
    print(model_name + ' total time: ', time.time() - start_time)
    print('Done.')

    # export models
    # dst_model_name = 'ssd_320_mnasnet_1a_dilated_cartoon_v9'
    dst_model_name = 'ssd_512_mnasnet_1a_dilated_guohui'
    dst_model_prefix = os.path.join('./checkpoints/', dst_model_name)
    net.export(dst_model_prefix, epoch=0)
