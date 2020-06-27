""" @package corner_pooling.py

    Include the center pooling & cascade corner pooling
    @author smh
    @date 2020.06.27
    @copyright
"""
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

from centernet_common import BasicConvBlock

class pool(gluon.HybridBlock):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.p2_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)

        self.p_conv1 = nn.Conv2D(dim, 3, 1, 1, use_bias=False, in_channels=128)
        self.p_bn1   = nn.BatchNorm(in_channels=dim)

        self.conv1 = nn.Conv2D(dim, 1, 1, 0, use_bias=False, in_channels=dim)
        self.bn1   = nn.BatchNorm(in_channels=dim)
        self.relu1 = nn.Activation(activation='relu')

        self.conv2 = BasicConvBlock(dim, 3, 1, 1, use_bias=False, in_c=dim)

        self.pool1 = pool1
        self.pool2 = pool2

        self.look_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.look_conv2 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.P1_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)
        self.P2_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)

    def hybrid_forward(self, F, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_cross(gluon.HybridBlock):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.p2_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)

        self.p_conv1 = nn.Conv2D(dim, 3, 1, 1, use_bias=False, in_channels=128)
        self.p_bn1   = nn.BatchNorm(in_channels=dim)

        self.conv1 = nn.Conv2D(dim, 1, 1, 0, use_bias=False, in_channels=dim)
        self.bn1   = nn.BatchNorm(in_channels=dim)
        self.relu1 = nn.Activation(activation='relu')

        self.conv2 = BasicConvBlock(dim, 3, 1, 1, use_bias=False, in_c=dim)

        self.pool1 = pool1
        self.pool2 = pool2
        self.pool3 = pool3
        self.pool4 = pool4

    def hybrid_forward(self, F, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class TopLeftPooling(gluon.HybridBlock):
    ''' Top Left Pooling (Cascade Top Corner Pooling)
    '''
    def __init__(self, dim):
        super(TopLeftPooling, self).__init__()
        self.p1_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.p2_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)

        self.p_conv1 = nn.Conv2D(dim, 3, 1, 1, use_bias=False, in_channels=128)
        self.p_bn1   = nn.BatchNorm(in_channels=dim)

        self.conv1 = nn.Conv2D(dim, 1, 1, 0, use_bias=False, in_channels=dim)
        self.bn1   = nn.BatchNorm(in_channels=dim)
        self.relu1 = nn.Activation(activation='relu')

        self.conv2 = BasicConvBlock(dim, 3, 1, 1, use_bias=False, in_c=dim)

        # self.pool1 = pool1            # Top Pooling
        # self.pool2 = pool2            # Left Pooling

        self.look_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.look_conv2 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.P1_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)
        self.P2_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)

    def hybrid_forward(self, F, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        # look_right   = self.pool2(look_conv1)
        look_right   = F.LeftRightPooling(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        # pool1        = self.pool1(P1_look_conv)
        pool1        = F.UpBottomPooling(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = F.UpBottomPooling(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = F.LeftRightPooling(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2
    

class BottomRightPooling(gluon.HybridBlock):
    def __init__(self, dim):
        super(BottomRightPooling, self).__init__()
        self.p1_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.p2_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)

        self.p_conv1 = nn.Conv2D(dim, 3, 1, 1, use_bias=False, in_channels=128)
        self.p_bn1   = nn.BatchNorm(in_channels=dim)

        self.conv1 = nn.Conv2D(dim, 1, 1, 0, use_bias=False, in_channels=dim)
        self.bn1   = nn.BatchNorm(in_channels=dim)
        self.relu1 = nn.Activation(activation='relu')

        self.conv2 = BasicConvBlock(dim, 3, 1, 1, use_bias=False, in_c=dim)

        # self.pool1 = pool1        # Bottom Pooling
        # self.pool2 = pool2        # Right Pooling

        self.look_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.look_conv2 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.P1_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)
        self.P2_look_conv = nn.Conv2D(128, 3, 1, 1, use_bias=False, in_channels=128)

    def hybrid_forward(self, F, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = F.RightLeftPooling(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = F.BottomUpPooling(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = F.BottomUpPooling(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = F.RightLeftPooling(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class CrossPooling(gluon.HybridBlock):
    ''' Center Pooling

        (1) Branch 1: Top & Bottom Pooling
        (2) Branch 2: Left & Right Pooling
        (3) Elemwise add of B1 & B2
    '''
    def __init__(self, dim):
        super(CrossPooling, self).__init__()
        self.p1_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)
        self.p2_conv1 = BasicConvBlock(128, 3, 1, 1, use_bias=False, in_c=dim)

        self.p_conv1 = nn.Conv2D(dim, 3, 1, 1, use_bias=False, in_channels=128)
        self.p_bn1   = nn.BatchNorm(in_channels=dim)

        self.conv1 = nn.Conv2D(dim, 1, 1, 0, use_bias=False, in_channels=dim)
        self.bn1   = nn.BatchNorm(in_channels=dim)
        self.relu1 = nn.Activation(activation='relu')

        self.conv2 = BasicConvBlock(dim, 3, 1, 1, use_bias=False, in_c=dim)

        # self.pool1 = pool1            # Top Pooling
        # self.pool2 = pool2            # Left Pooling
        # self.pool3 = pool3            # Bottom Pooling
        # self.pool4 = pool4            # Right Pooling

    def hybrid_forward(self, F, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = F.UpBottomPooling(p1_conv1)
        pool1    = F.BottomUpPooling(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = F.LeftRightPooling(p2_conv1)
        pool2    = F.RightLeftPooling(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

if __name__ == '__main__':
    data = nd.random.uniform(0.0, 1.0, (1, 128, 32, 32))
    # net = TopLeftPooling(128)
    # net = BottomRightPooling(128)
    net = CrossPooling(128)
    net.initialize()
    output = net(data)
    print('shape of output: ', output.shape)
