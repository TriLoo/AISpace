# -*- coding: utf-8 -*-

import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, autograd
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import utils_old as utils
from utils import voc_map_ori
import common
# classes = common.mapped_cls
# classes = ['bdbk']
classes = ['guohui']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--model_name', type=str, default='ssd_320_mnasnet_1a_test')
    parser.add_argument('--params', type=str, default='./scripts/mnasnet1a_ssd_320_large_head/mnasnet1a_ssd_320_large_head_best.params')
    parser.add_argument('--img_root', type=str, default='./datasets/cocovoc_cartoon')
    parser.add_argument('--data_shape', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mx', action='store_true')
    args = parser.parse_args()

    return args


def get_dir_imgs(img_dir):
    img_exts = ['.jpg', '.jpeg', '.png', '.tif']
    ret_imgs = []
    for root, folder, files in os.walk(img_dir):
        if len(files) < 1:
            continue
        for img_file in files:
            img_file_ext = os.path.splitext(img_file)[-1]
            if img_file.startswith('.'):
                continue
            if img_file_ext in img_exts:
                ret_imgs.append(os.path.join(root, img_file))

    return ret_imgs


def eval_net(net, val_data, eval_metric, ctx):
    eval_metric.reset()
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)

    return eval_metric.get()


def get_mod(model_prefix, ctx, batch_size=1, epoch=0):
    sym, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    mod = mx.mod.Module(sym, data_names=['data'], label_names=None, context=ctx)
    mod.bind(data_shapes=[('data', (batch_size, 3, 320, 320))], for_training=False)
    # mod.bind(data_shapes=[('data', (batch_size, 3, 512, 512))], for_training=False)
    mod.set_params(args, auxs)

    return mod

def eval_mod_dataloader(model_prefix, train_data, ctx, batch_size=1, epoch=0):
    mod = get_mod(model_prefix, ctx, batch_size, epoch)
    # eval_metric = gcv.utils.metrics.VOC07MApMetric(class_names=classes)
    # eval_metric = voc_map.VOC07MApMetric(class_names=classes, roc_output_path='../datasets/roc_results')
    # eval_metric = eval_metric_2.VOC07MApMetric(class_names=classes, roc_output_path='../datasets/roc_results')
    eval_metric.reset()

    with tqdm(total=len(train_data)) as pbar:
        start = time.time()
        for ib, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                mod.predict(x)
                ids, scores, bboxes = mod.get_outputs()
                det_ids.append(ids)
                det_scores.append(scores)
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

                # preds = nd.concat(ids, scores, bboxes, dim=-1)
                # gt_cids = y.slice_axis(axis=-1, begin=4, end=5)
                # gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
                # labels = nd.concat(gt_cids, gt_bboxes, dim=-1)
                # eval_metric.update(labels, preds)

            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            # eval_metric.update(pred_list, label_list)
            # pbar.update(batch[0].shape[0])
            pbar.update(1)

        end = time.time()
        speed = len(train_data) / (end - start)
        print('Throughput is %f img/sec.'% speed)

    return eval_metric.get()


def get_argsort(a, axis=1, kind='stable'):
    a_np = a.asnumpy()
    a_idx = np.argsort(a_np, axis, kind)

    return a_idx


def apply_argsort(a, idx, axis=1):
    # print('shape of idx:  ', idx.shape)
    for i in range(idx.shape[0]):
        sort_idx = idx[i, :, 0]
        tmp = a[i, sort_idx, :]
        a[i, :, :] = tmp

    return a


def eval_mod_mx_dataloader(model_prefix, train_data, ctx, batch_size=1, epoch=0):
    mod = get_mod(model_prefix, ctx, batch_size, epoch)
    eval_metric = gcv.utils.metrics.VOC07MApMetric(class_names=classes)
    eval_metric.reset()

    with tqdm(total=len(train_data)) as pbar:
        start = time.time()
        for ib, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                mod.predict(x)
                preds = mod.get_outputs()
                ids = preds[0][:, :600, :1]
                scores = preds[0][:, :600, 1:2]
                bboxes = preds[0][:, :600, 2:]

                obj_mask = mx.nd.where(ids >= 0, mx.nd.ones_like(ids) * -1, mx.nd.zeros_like(ids))
                idx = get_argsort(obj_mask)
                ids = apply_argsort(ids, idx)[:, :100, :]
                scores = apply_argsort(scores, idx)[:, :100, :]
                bboxes = apply_argsort(bboxes, idx)[:, :100, :]

                det_ids.append(ids)
                det_scores.append(scores)
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            # pbar.update(batch[0].shape[0])
            pbar.update(1)

        end = time.time()
        speed = len(train_data) / (end - start)
        print('Throughput is %f img/sec.'% speed)

    return eval_metric.get()


def eval_mod_img(model_prefix, img_file, ctx, epoch=0, show_top10=False):
    mod = get_mod(model_prefix, ctx, batch_size=1, epoch=epoch)
    data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=320)
    # data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=512)
    print('shape of data: ', data.shape)
    print('shape of img: ', img.shape)

    mod.predict(data)
    ids, scores, bboxes = mod.get_outputs()
    if show_top10:
        print(ids[0][:10, :])
        print(scores[0][:10, :])
        print(bboxes[0][:10, :])

    gcv.utils.viz.plot_bbox(img, bboxes[0], scores[0], ids[0], thresh=0.3, class_names=classes)

    plt.savefig('./badcases/results.png')
    plt.show()

    print('Done.')


def eval_mod_mx_img(model_prefix, img_file, ctx, epoch=0):
    mod = get_mod(model_prefix, ctx, batch_size=1, epoch=epoch)
    data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=320)

    mod.predict(data)
    preds = mod.get_outputs()

    ids = preds[0][:, :100, 0]
    scores = preds[0][:, :100, 1]
    bboxes = preds[0][:, :100, 2:]

    gcv.utils.viz.plot_bbox(img, bboxes[0], scores[0], ids[0], thresh=0.3, class_names=classes)
    plt.savefig('./badcases/results.png')
    plt.show()

    print('Done.')


def eval_mod_img_dir(model_prefix, img_dir, dst_dir, ctx, epoch=0):
    mod = get_mod(model_prefix, ctx, batch_size=1, epoch=epoch)

    # get the images
    imgs_list = get_dir_imgs(img_dir)
    print('input img nums: ', len(imgs_list))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # do the inference
    valid_img_nums = 0
    for img_file in imgs_list:
        print('current imgfile: ', img_file)
        img_file_name = os.path.split(img_file)[-1]
        img_name = os.path.splitext(img_file_name)[0]
        dst_img_file = os.path.join(dst_dir, img_name + '_res.jpeg')
        try:
            data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=320)
            # data, img = gcv.data.transforms.presets.ssd.load_test(img_file, short=512)
        except:
            print('Input img file {} read failed.'.format(img_file))
            continue
        data = data.copyto(ctx)

        mod.predict(data)
        ids, scores, bboxes = mod.get_outputs()
        if scores[0, 0, 0] < 0.8:
            continue

        # gcv.utils.viz.plot_bbox(img, bboxes[0], scores[0], ids[0], thresh=0.3, class_names=classes)
        gcv.utils.viz.plot_bbox(img, bboxes[0], scores[0], ids[0], thresh=0.8, class_names=classes)
        plt.savefig(dst_img_file)

        valid_img_nums += 1

    print('Valid img nums: ', valid_img_nums)
    print('Done.')


if __name__ == '__main__':
    args = parse_args()

    # ----------- ************* -------------
    # batch evaluation
    # ----------- ************* -------------

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # test_set = utils.VOCDetection(root=args.img_root, splits=[(2007, 'test')])
    # test_set = utils.VOCDetection(root=args.img_root, splits=[(2007, 'test'), (2012, 'test')])
    # val_data = utils.get_evaldataloader(test_set, data_shape=args.data_shape, batch_size=args.batch_size, num_workers=args.num_workers)

    model_name = args.model_name
    # eval_metric = gcv.utils.metrics.VOC07MApMetric(class_names=classes)
    # eval_metric = voc_map.VOC07MApMetric(class_names=classes, roc_output_path='../datasets/roc_results')
    # eval_metric = eval_metric_2.VOC07MApMetric(class_names=classes, roc_output_path='../datasets/roc_results')
    eval_metric = voc_map_ori.VOC07MApMetric(class_names=classes, roc_output_path='../datasets/roc_results/v7_results/roc_results_mx', score_thresh=0.0)

    # -------------- using gcv net ----------------
    # net = gcv.model_zoo.get_model(model_name, classes=classes)
    # net.initialize(force_reinit=True)
    # if args.params is not None:
    #     # net.load_parameters(args.params, allow_missing=True, ignore_extra=True)
    #     net.load_parameters(args.params)
    #     print('Loading weight done.')
    # k, v = eval_net(net, val_data, eval_metric, ctx)

    # -------------- using module ---------------
    # if args.is_mx:
    #     print('Using mx head ...')
    #     k, v = eval_mod_mx_dataloader(model_name, val_data, ctx, args.batch_size)
    # else:
    #     print('Using gluoncv head ...')
    #     k, v = eval_mod_dataloader(model_name, val_data, ctx, args.batch_size)

    # for x, y in zip(k, v):
    #     print(x, y)

    #### Model Prefix
    ####
    #### #### #### #### ####

    # ctx = mx.cpu(2)
    ctx = mx.gpu(1)
    #### 1. original
    # model_prefix = './models/mnasnet1a_ssd_320_nobn_mx'
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated'

    #### 2. original + cartoon models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon'

    #### 3. original + cartoon v3 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_v3'

    #### 4. original + cartoon v5 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_v5'

    #### 5. original + cartoon v5 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_test'

    #### 6. original + cartoon v5 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_v6'

    #### 7. original + cartoon v7 temp models, current work in ai crop!!!
    # model_prefix = './checkpoints/ssd_320_mnasnet_1a_dilated_cartoon_v7_temp'

    #### 8. original + cartoon v8 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_v8'

    #### 9. original + cartoon v9 models
    # model_prefix = './models/ssd_320_mnasnet_1a_dilated_cartoon_v9'

    #### #### #### ####
    #### 0. used for bdbk logo detection
    # model_prefix = './models/ssd_512_mnasnet_1a_dilated_bdbk'

    #### #### #### ####
    #### 0. used for guohui detection
    model_prefix = './checkpoints/ssd_512_mnasnet_1a_dilated_guohui'


    # ----------- ************* -------------
    #  Use image dir as input for batch test, no display
    # ----------- ************* -------------

    #### input & output image dir name
    # img_dir = './badcases/testdata_20200410'
    # dst_dir = './badcases/testdata_20200410_res'

    # img_dir = './badcases/baidu_tuku'
    # dst_dir = './badcases/baidu_tuku_res'

    # img_dir = './badcases/baidu_tuku'
    # dst_dir = './badcases/baidu_tuku_res'

    # img_dir = '/search/odin/zhangjun/val/top100k'
    # dst_dir = './badcases/bdbk_top100k_res'

    # img_dir = './badcases/testdata_20200514'
    # dst_dir = './badcases/testdata_20200514_res'

    # img_dir = './badcases/testdata_guohui'
    # dst_dir = './badcases/testdata_guohui_res'

    img_dir = '/search/odin/songminghui/datasets/Yingxiao/aiaudit/imgs'
    dst_dir = './badcases/testdata_guohui_res'

    eval_mod_img_dir(model_prefix, img_dir, dst_dir, ctx)

    # ----------- ************* -------------
    # Show single image using mod
    # ----------- ************* -------------

    #### image filename
    # img_file = './badcase20191108/75.jpg'
    # img_file = './badcase20191108/aicrop_badcase20191111.jpg'
    # img_file = './badcase20191108/QQ20191112-0.jpg'
    # img_file = './badcase20191108/03644.jpg'
    # img_file = './badcase20191108/09327.jpg'
    # img_file = './badcase20191108/75_1.jpg'
    # img_file = './badcase20191108/QQ20191114-1.jpg'
    # img_file = './badcase20191108/75_2.jpeg'
    # img_file = './badcase20191108/wangchuanjun.jpeg'
    # img_file = './badcase20191108/QQ20191118-0.jpg'
    # img_file = './badcase20191120/QQ20191120-0.jpg'
    # img_file = './badcase20191120/75_84.jpeg'
    # img_file = './badcases/testdata_20191122/2019_11_25_d94ef13c0a584bad947a6f59a75a483d.png'
    # img_file = './badcases/all_badcases/df92-iieqapu5371410.jpg'
    # img_file = './badcases/testdata_20191126/d24116a3080220f3255ec889ee31c8eb.jpeg'
    # img_file = './badcases/testdata_20191129/image_413017_1574876359025.jpeg'
    # img_file = './badcases/testdata_20200310/img_420957_1583627725173.jpeg'
    # img_file = './badcases/testdata_20200402/389c-irpunai5537606.jpg'
    # img_file = './badcases/baidu_tuku/0b46f21fbe096b633ea81ee005338744eaf8acc9.jpg'
    # img_file = './badcases/baidu_tuku/bdbk_tmp.jpg'
    img_file = './badcases/WechatIMG27.jpeg'

    ctx = mx.cpu(2)

    # eval_mod_mx_img(model_prefix, img_file, ctx)
    # eval_mod_img(model_prefix, img_file, ctx, show_top10=True)
