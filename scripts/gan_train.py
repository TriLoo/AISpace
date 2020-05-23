# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import autograd, nd, gluon
from mxboard import SummaryWriter

import os
import cv2
import numpy as np
import time
import logging

import pix2pix_mx
import utils

train_dir = './datasets/facades/train'
test_dir = './datasets/facades/test'
val_dir = './datasets/facades/val'


def loadDataset(data_dir, bs=4, data_shape=256, is_reverse=False):
    img_file_list = []
    ## get all the image files
    for root, _, img_files in os.walk(data_dir):
        for img_file in img_files:
            if img_file.endswith('.jpg'):
                img_file_list.append(os.path.join(root, img_file))
    ## read the images & labels
    img_list = []
    label_list = []
    for img_file in img_file_list:
        # img = cv2.imread(img_file, -1)
        # img = cv2.resize(img, (data_shape * 2, data_shape))             # (new_w, new_h)
        # # convert to [-1, 1]
        # img = img.astype(np.float32) / 127.5 - 1
        # # split img data & label
        # img_list.append(nd.array(img[:, 0:data_shape, :]).expand_dims(axis=0).transpose((0, 3, 1, 2)))
        # label_list.append(nd.array(img[:, data_shape:, :]).expand_dims(axis=0).transpose((0, 3, 1, 2)))

        img_arr = mx.image.imread(img_file).astype(np.float32) / 127.5 - 1
        img_arr = mx.image.imresize(img_arr, data_shape * 2, data_shape)
        img_arr_in, img_arr_label = [mx.image.fixed_crop(img_arr, 0, 0, data_shape, data_shape), 
                                        mx.image.fixed_crop(img_arr, data_shape, 0, data_shape, data_shape)]
        img_arr_in, img_arr_label = [nd.transpose(img_arr_in, (2,0,1)),
                                    nd.transpose(img_arr_label, (2,0,1))]
        img_arr_in, img_arr_label = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                    img_arr_label.reshape((1,) + img_arr_label.shape)]
        img_list.append(img_arr_label if is_reverse else img_arr_in)
        label_list.append(img_arr_in if is_reverse else img_arr_label)
    
    ## generate data iterator
    # if is_reverse:
    #     return mx.io.NDArrayIter(nd.concat(*label_list, dim=0), nd.concat(*img_list, dim=0), batch_size=bs, shuffle=True, data_name='facades_data', label_name='facades_label')
    # else:
    #     return mx.io.NDArrayIter(nd.concat(*img_list, dim=0), nd.concat(*label_list, dim=0), batch_size=bs, shuffle=True, data_name='facades_data', label_name='facades_label')
    return mx.io.NDArrayIter(nd.concat(*img_list, dim=0), nd.concat(*label_list, dim=0), batch_size=bs, shuffle=False, data_name='facades_data', label_name='facades_label')


img_wd = 256
img_ht = 256
def load_data(path, batch_size, is_reversed=False):
    img_in_list = []
    img_out_list = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith('.jpg'):
                continue
            img = os.path.join(path, fname)
            img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
            img_arr = mx.image.imresize(img_arr, img_wd * 2, img_ht)
            # Crop input and output images
            img_arr_in, img_arr_out = [mx.image.fixed_crop(img_arr, 0, 0, img_wd, img_ht),
                                       mx.image.fixed_crop(img_arr, img_wd, 0, img_wd, img_ht)]
            img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2,0,1)),
                                       nd.transpose(img_arr_out, (2,0,1))]
            img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                       img_arr_out.reshape((1,) + img_arr_out.shape)]
            img_in_list.append(img_arr_out if is_reversed else img_arr_in)
            img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    # return mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0), nd.concat(*img_out_list, dim=0)],
    #                          batch_size=batch_size)
    return mx.io.NDArrayIter(data=nd.concat(*img_in_list, dim=0), label=nd.concat(*img_out_list, dim=0), batch_size=batch_size, shuffle=False, data_name='facades_data', label_name='facades_label')


## configs
ctx = mx.gpu(4)
bs = 10
l1_lambda = 1e2
log_dir = './checkpoints/facades_pix2pix/20200120/'
ckpt_fmt = os.path.join(log_dir, 'params/{}_epoch{}.params')
save_img_fmt = os.path.join(log_dir, 'ckpt_imgs/epoch{}_idx_{}.jpg')
log_iter_intervals = 20
save_img_intervals = 10

## datasets
train_iter = loadDataset(train_dir, bs=bs, is_reverse=True)
val_iter = loadDataset(val_dir, bs=bs, is_reverse=True)
# train_iter = load_data(train_dir, bs, is_reversed=True)
# val_iter = load_data(val_dir, bs, is_reversed=True)

## networks
netG = pix2pix_mx.UnetGenerator(3, 3, 8, use_dropout=True)
netD = pix2pix_mx.NLayerDiscriminator(6)
# netD = pix2pix_mx.Discriminator(in_channels=6)
utils.init_network(netG)
utils.init_network(netD)
netG.collect_params().reset_ctx(ctx)
netD.collect_params().reset_ctx(ctx)


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()

    return ((pred > 0.5) == label).mean()


def eval(epch, iter_step=None):
    val_iter.reset()
    for iter_step, databatch in enumerate(val_iter):
        data = databatch.data[0].as_in_context(ctx)
        pred = netG(data)

        if (iter_step + 1) % save_img_intervals:
            utils.save_img(pred, save_img_fmt, epch, is_rgb=False)


def train(epochs):
    gan_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    l1_loss = gluon.loss.L1Loss()

    trainer_G = gluon.Trainer(netG.collect_params(), 'adam', optimizer_params={'learning_rate' : 0.0002, 'beta1' : 0.5, 'beta2' : 0.999})
    trainer_D = gluon.Trainer(netD.collect_params(), 'adam', optimizer_params={'learning_rate' : 0.0002, 'beta1' : 0.5, 'beta2' : 0.999})

    ## config the log file
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    logger.addHandler(fh)

    sw = SummaryWriter(logdir=os.path.join(log_dir, 'train_sw'))
    batch_len = train_iter.num_data // train_iter.batch_size

    image_pool = utils.ImagePool(50)

    global_step = 0
    for epch in range(epochs):
        train_iter.reset()
        epch_time = time.time()
        batch_time = time.time()
        for iter_step, databatch in enumerate(train_iter):
            data = databatch.data[0].as_in_context(ctx)
            label = databatch.label[0].as_in_context(ctx)

            ## train netD
            pred = netG(data)
            # fake_data =nd.concat(data, pred, dim=1)
            # fake_data = image_pool.fetch_img(fake_data)
            fake_data = image_pool.fetch_img(nd.concat(data, pred, dim=1))
            with autograd.record():
                # fake
                pred_fake = netD(fake_data)
                fake_label = nd.zeros_like(pred_fake)
                loss_fake = gan_loss(pred_fake, fake_label).sum()
                # real
                real_data = nd.concat(data, label, dim=1)
                pred_real = netD(real_data)
                real_label = nd.ones_like(pred_real)
                loss_real = gan_loss(pred_real, real_label).sum()

                loss_D = (loss_real + loss_fake) * 0.5
                loss_D.backward()
            trainer_D.step(data.shape[0])
            sw.add_scalar('lossD', loss_D.asscalar(), global_step)

            ## train netG
            with autograd.record():
                pred = netG(data)
                in_data = nd.concat(data, pred, dim=1)
                pred_real = netD(in_data)
                pred_label = nd.ones_like(pred_real)

                ganloss_g = gan_loss(pred_real, pred_label)
                l1loss_g = l1_loss(pred, label)
                loss_G = ganloss_g + l1loss_g * l1_lambda
                loss_G = loss_G.sum()
                loss_G.backward()
            trainer_G.step(data.shape[0])
            sw.add_scalar('lossG', loss_G.asscalar(), global_step)

            ## do the checkpoints during intra epoch
            if (iter_step + 1) % log_iter_intervals == 0:
                logger.info('[Epoch {}][Iter {}] Done., Speed: {:.4f} sample / s'.format(str(epch), str(iter_step), data.shape[0] / (time.time() - batch_time)))

            batch_time = time.time()
            global_step += 1

        ## do the evaluation after every epoch
        fake_img = pred[0]
        img_arr = (fake_img - mx.nd.min(fake_img)) / (mx.nd.max(fake_img) - mx.nd.min(fake_img))
        # img_arr = img_arr[::-1, :, :]
        sw.add_image('generated image', img_arr)
        eval(epch)

        ## do the checkpoints inter epochs
        netG.save_parameters(ckpt_fmt.format('netG', str(epch)))
        netD.save_parameters(ckpt_fmt.format('netD', str(epch)))

        logger.info('[Epoch {}] Done. Cost: {:.4f} s'.format(str(epch), time.time() - epch_time))


if __name__ == '__main__':
    epochs = 100
    train(epochs)
    # for databatch in train_iter:
    #     data = databatch.data[0]
    #     utils.save_img(data, '{}_in_img_{}.jpg', 0)
    #     label = databatch.label[0]
    #     utils.save_img(label, '{}_label_{}.jpg', 0)
    #     break
