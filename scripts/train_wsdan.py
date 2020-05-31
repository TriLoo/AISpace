""" @package train_wsdan.py
    implementation of <See Better Before Looking Closer: ...> based on ResNet34
"""
# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, '..')

import mxnet as mx
from mxnet import gluon, autograd
import gluoncv as gcv
from gluoncv import model_zoo
import random

import time
import logging
import argparse
import math
import numpy as np
from utils import load_rec_datasets

EPSILON = 1e-12

def batch_argument(images, attention_map, mode='crop', theta=0.5, pading_ratio=0.1):
    batches, _, imgH, imgW = images.shape
    if mode == 'crop':
        crop_images = []
        for batch_idx in range(batches):
            atten_map = attention_map[batch_idx : batch_idx + 1]
            if isinstance(theta, (list, tuple)):
                theta_c = random.uniform(*theta) * (atten_map.max() - atten_map.min()) + atten_map.min()
            else:
                theta_c = theta * (atten_map.max() - atten_map.min()) + atten_map.min()

            # TODO: change the implement to not use GGPU -> CPU copy and get the outest box position
            crop_mask = mx.nd.contrib.BilinearResize2D(atten_map, height=imgH, width=imgW)[0, 0, ...] >= theta_c
            # print('shape of crop_mask: ', crop_mask.shape)
            crop_mask_sumrow = mx.nd.sum(crop_mask, axis=1)
            crop_mask_row = mx.nd.where(crop_mask_sumrow > 0, mx.nd.arange(0, crop_mask.shape[0], ctx=crop_mask_sumrow.context), mx.nd.zeros_like(crop_mask_sumrow))
            crop_mask_sumcol = mx.nd.sum(crop_mask, axis=0)
            crop_mask_col = mx.nd.where(crop_mask_sumcol > 0, mx.nd.arange(0, crop_mask.shape[1], ctx=crop_mask_sumcol.context), mx.nd.zeros_like(crop_mask_sumcol))

            height_min = max(min(int(crop_mask_row.min().asscalar() - pading_ratio * imgH), int((1 - 2 * pading_ratio) * imgH)), 0)
            height_max = min(max(int(crop_mask_row.max().asscalar() - pading_ratio * imgH), int(pading_ratio * imgH)), imgH)
            width_min = max(min(int(crop_mask_col.min().asscalar() -  pading_ratio * imgW), int((1 - 2 * pading_ratio) * imgW)), 0)
            width_max = min(max(int(crop_mask_col.max().asscalar() - pading_ratio * imgW), int(pading_ratio * imgW)), imgW)

            # crop_mask = mx.nd.contrib.BilinearResize2D(atten_map, height=imgH, width=imgW).asnumpy() >= theta_c.asscalar()
            # nonzero_idx = np.nonzero(crop_mask[0, 0, ...])
            # # print('nonzero_idx[1]: ', nonzero_idx[1])
            # # print('max of nonzero_idx[1]: ', nonzero_idx[1].max())
            # height_min = max(min(int(nonzero_idx[0].min() - pading_ratio * imgH), int((1 - 2 * pading_ratio) * imgH)), 0)
            # height_max = min(max(int(nonzero_idx[0].max() - pading_ratio * imgH), int(pading_ratio * imgH)), imgH)
            # width_min = max(min(int(nonzero_idx[1].min() -  pading_ratio * imgW), int((1 - 2 * pading_ratio) * imgW)), 0)
            # width_max = min(max(int(nonzero_idx[1].max() - pading_ratio * imgW), int(pading_ratio * imgW)), imgW)
            # print('current h_min: ', height_min)
            # print('current h_max: ', height_max)
            # print('current w_min: ', width_min)
            # print('current w_max: ', width_max)

            crop_images.append(mx.nd.contrib.BilinearResize2D(images[batch_idx : batch_idx + 1, :, height_min : height_max, width_min:width_max], height=imgH, width=imgW))
        crop_images = mx.nd.concat(*crop_images, dim=0)

        return crop_images
    elif mode == 'drop':
        drop_masks = []
        for batch_idx in range(batches):
            atten_map = attention_map[batch_idx : batch_idx + 1]
            if isinstance(theta, (list, tuple)):
                theta_d = random.uniform(*theta) * (atten_map.max() - atten_map.min()) + atten_map.min()
            else:
                theta_d = theta * (atten_map.max() - atten_map.min()) + atten_map.min()
            
            drop_masks.append(mx.nd.contrib.BilinearResize2D(atten_map, height=imgH, width=imgW) < theta_d)
        drop_masks = mx.nd.concat(*drop_masks, dim=0)
        drop_images = images * drop_masks

        return drop_images

    else:
        raise ValueError("Expected mode in ['crop', 'drop'], but got %s " % mode)

def save_params(net, best_acc, current_acc, epoch, save_interval, prefix):
    current_acc = float(current_acc)
    if current_acc > best_acc[0]:
        best_acc[0] = current_acc
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_acc))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_acc))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_acc))

def evaluate(net, eval_data, ctx):
    acc_metric = mx.metric.Accuracy()
    ce_metric = mx.metric.Loss('CrossEntropy')
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    acc_metric.reset()
    ce_metric.reset()
    for i, batch in enumerate(eval_data):
        # data = gluon.utils.split_and_load(batch[0], ctx, batch_axis=0, even_split=False)
        # label = gluon.utils.split_and_load(batch[1], ctx, batch_axis=0,even_split=False)
        data = gluon.utils.split_and_load(batch[0], ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx, batch_axis=0)
        # outputs = [net(X) for X in data]
        outputs = [net(X)[0] for X in data]
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        test_loss = sum([l.mean().asscalar() for l in loss]) / len(loss)
        acc_metric.update(label, outputs)
        ce_metric.update(0, [mx.nd.array([test_loss])])

    return acc_metric.get(), ce_metric.get()

def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, mx.nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value = eta / classes)
        y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes )
        res.append(lam * y1 + (1 - lam) * y2)

    return res

def smooth_label(label, classes, eta=0.1):
    if isinstance(label, mx.nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value = eta / classes)
        smoothed.append(res)
    return smoothed



def train(net, train_data, eval_data, ctx, args, batch_size, epoches=1):
    net.hybridize(static_alloc=True, static_shape=True)
    net.collect_params().reset_ctx(ctx)

    lr = args.lr
    momentum = args.momentum
    wd = args.wd
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_iter.split(',') if ls.strip()])

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum':momentum, 'wd':wd})
    if args.use_mixup or args.label_smoothing:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)   # Used for smooth label
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()
    center_loss = gluon.loss.L2Loss()
    ce_metric = mx.metric.Loss('CrossEntropy')

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_acc = [0]

    # feature_center = mx.nd.zeros((args.class_num * batch_size, 32 * 512))
    feature_center = mx.nd.zeros((batch_size, args.class_num, 32 * 512))
    for epoch in range(args.start_epoch, args.epochs):
        feature_center_lst = gluon.utils.split_and_load(feature_center, ctx_list=ctx, batch_axis=0)
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info('[Epoch {}] Set learning rate to {}'.format(epoch, new_lr))
        logger.info('[Epoch {}] Set learning rate to {}'.format(epoch, trainer.learning_rate))

        ce_metric.reset()
        tic = time.time()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label_ori = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            if args.label_smoothing:
                label = smooth_label(label_ori, classes=args.class_num)
            else:
                label = label_ori

            with autograd.record():
                loss = []
                for X, y, feat_center, y_hard in zip(data, label, feature_center_lst, label_ori):
                    # print('shape of X: ', X.shape)
                    # print('shape of y_hard: ', y_hard.shape)
                    # print('shape of feat_center: ', feat_center.shape)
                    # print('current y: ', y_hard)
                    y_hard = y_hard.squeeze().astype("int32")
                    pred, feat, atten = net(X)
                    cls_loss_raw = L(pred, y)
                    # crop
                    with autograd.pause():
                        crop_img = batch_argument(X, atten[:, :1, :, :], 'crop', [0.4, 0.6])
                    pred_crop, _, _ = net(crop_img)
                    cls_loss_crop =  L(pred_crop, y)
                    # drop
                    with autograd.pause():
                        drop_img = batch_argument(X, atten[:, 1:, :, :], 'drop', [0.2, 0.5])
                    pred_drop, _, _ = net(drop_img)
                    cls_loss_drop = L(pred_drop, y)
                    # center
                    with autograd.pause():
                        # udpate the feature center
                        # I cannot figure out how mx.nd.pick() used to index along the axis=1
                        feat_center_batch = []
                        for i in range(y_hard.shape[0]):
                            feat_center_single = feat_center[i, y_hard[i], :] / (mx.nd.norm(feat_center[i, y_hard[i], :], axis=-1) + EPSILON)
                            feat_center[i, y_hard[i], :] += 5e-2 * (feat[i, :] - feat_center_single)
                            feat_center_batch.append(feat_center_single)
                        feat_center_batch = mx.nd.concat(*feat_center_batch, dim=0)

                        # print('shape of y_hard: ', y_hard.shape)
                        # print('shape of feat: ', feat.shape)
                        # feat_center_batch = feat_center[:, y_hard] / (mx.nd.norm(feat_center[:, y_hard], axis=-1) + EPSILON)
                        # print('shape of feat_center_batch: ', feat_center_batch.shape)
                        # feat_center[:, y_hard] += 5e-2 * (feat - feat_center_batch)
                    reg_loss = center_loss(feat_center_batch, feat)
                    
                    curr_loss = cls_loss_raw + cls_loss_crop + cls_loss_drop + reg_loss
                    loss.append(curr_loss)
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss = sum([l.mean().asscalar() for l in loss]) / len(loss)
            ce_metric.update(0, [mx.nd.array([train_loss])])
            if args.log_interval and not (i + 1) % args.log_interval:
                name, acc = ce_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed {:.3f} samples/sec, {}={:.3f}'.format(epoch, i, batch_size/(time.time() - btic), name, acc), )
            btic = time.time()

        name, train_acc = ce_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}'.format(epoch, (time.time() - tic), name, train_acc))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            (name1, test_acc), (name2, test_ce) = evaluate(net, eval_data, ctx)
            logger.info('[Epoch {}] Validation loss: {}, Validation acc: {} \n'.format(epoch, test_ce, test_acc))
            current_acc = test_acc
        else:
            current_acc = 0.0
        save_params(net, best_acc, current_acc, epoch, args.save_interval, args.save_prefix)




def parse_args():
    parser = argparse.ArgumentParser(description='Finetune own SSD networks.')
    parser.add_argument('--model_name', type=str, default='mobilenetv1_0.75')
    parser.add_argument('--params', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/VOCdevkit')
    parser.add_argument('--classes', nargs='*')
    parser.add_argument('--data_shape', type=int, default=224,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num_workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr_decay_iter', type=str, default='1100,1700',
                        help='epochs at which learning rate decays. default is 1100, 1700.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save_prefix', type=str, default='ssd_mobilenet',
                        help='Saving parameter prefix')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--mixup_off_epoch', type=int, default=0)
    parser.add_argument('--class_num', type=int, default=11)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # prepare context
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_shape = args.data_shape

    train_set_path = os.path.join(dataset_dir, 'lq_imgs_norm_nausea_train_add_repeat.rec')
    test_set_path = os.path.join(dataset_dir, 'lq_imgs_norm_nausea_test.rec')

    print(data_shape)
    train_data = load_rec_datasets.load_train_datasets(train_set_path, batch_size, data_shape=data_shape, num_workers=num_workers)
    test_data = load_rec_datasets.load_eval_datasets(test_set_path, batch_size, data_shape=data_shape, num_workers=num_workers)

    model_name = args.model_name
    net = model_zoo.get_model(model_name, pretrained=True, batch_size=batch_size // len(ctx))
    net.dense = mx.gluon.nn.Dense(args.class_num)
    net.initialize()

    if args.params is not None:
        weight_file = args.params
        net.load_parameters(weight_file)

    train(net, train_data, eval_data=test_data, ctx=ctx, args=args, batch_size=batch_size, epoches=0)
