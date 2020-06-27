""" @package centernet_common.py

    @author smh
    @date 2020.06.27
    @copyright
"""
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

__all__ = ['BasicConvBlock', 'ResConvBlock']

class BasicConvBlock(gluon.HybridBlock):
    def __init__(self, c_out, kernel_size, strides, padding, use_bias=True, in_c=0, use_bn=True, use_act=True, **kwargs):
        super(BasicConvBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.feat = nn.HybridSequential()
            self.feat.add(
                nn.Conv2D(c_out, kernel_size, strides, padding, use_bias=use_bias, in_channels=in_c),
            )
            if use_bn:
                self.feat.add(
                    nn.BatchNorm(in_channels=c_out)
                )
            if use_act:
                self.feat.add(nn.Activation(activation='relu'))

    def hybrid_forward(self, F, x):
        feat = self.feat(x)

        return feat


class ResConvBlock(gluon.HybridBlock):
    def __init__(self, c_out, kernel_size=3, strides=1, padding=1, use_bias=False, in_c=0, **kwargs):
        super(ResConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = BasicConvBlock(c_out, kernel_size, strides, padding, use_bias, in_c)
            self.conv2 = BasicConvBlock(c_out, 3, 1, 1, use_bias=False, in_c=c_out, use_act=False)
            if strides != 1 or in_c != c_out:
                self.skip = BasicConvBlock(c_out, 1, 1, 0, use_bias=False, in_c=in_c, use_act=False)
            else:
                self.skip = gluon.contrib.nn.Identity()
            self.act = nn.Activation(activation='relu')

    def hybrid_forward(self, F, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        skip = self.skip(x)
        output = self.act(skip + feat)

        return output


if __name__ == '__main__':
    pass

