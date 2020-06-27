""" @package ghostnet.py
  implementation of "GhostNet: More Features from Cheap Operations"

  @author smh
  @date 2020.06.06
  @copyright Copyright (c) 2020 MIT
"""

# -*- coding: utf-8 -*-

__all__ = ['GhostNet', 'ghostnet512']

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn


class BasicConv2D(gluon.HybridBlock):
    def __init__(self, out_channels, ksize, stride, padding, use_act=True, in_channels=0, use_bias=True, groups=1, **kwargs):
        super(BasicConv2D, self).__init__(**kwargs)

        with self.name_scope():
            self.feat = nn.HybridSequential()
            self.feat.add(nn.Conv2D(out_channels, ksize, stride, padding, groups=groups, in_channels=in_channels, use_bias=use_bias))
            self.feat.add(nn.BatchNorm(in_channels=out_channels))
            if use_act:
                self.feat.add(nn.Activation(activation='relu'))

    def hybrid_forward(self, F, x):
        feat = self.feat(x)
        return feat

class GhostModule(gluon.HybridBlock):
    def __init__(self, out_channels, ksize, stride, padding, s=2, proj_size=3, proj_padding=1, use_act=True, in_channels=0, use_bias=True, **kwargs):
        super(GhostModule, self).__init__(**kwargs)
        m = out_channels // s
        proj_out_channels = int(out_channels - m)
        self.use_act = use_act
        with self.name_scope():
            self.feat = nn.HybridSequential()
            # self.feat.add(nn.Conv2D(m, ksize, stride, padding, in_channels=in_channels, use_bias=use_bias))
            self.feat.add(BasicConv2D(m, ksize, stride, padding, in_channels=in_channels, use_bias=use_bias))
            self.proj = nn.HybridSequential()
            # self.proj.add(nn.Conv2D(proj_out_channels, proj_size, 1, proj_padding, in_channels=in_channels, use_bias=use_bias, groups=m))
            self.proj.add(BasicConv2D(proj_out_channels, proj_size, 1, proj_padding, in_channels=m, use_bias=use_bias, groups=m))
            # if use_act:
            #     self.act = nn.HybridSequential()
            #     self.act.add(nn.BatchNorm())
            #     self.act.add(nn.Activation(activation='relu'))

    def hybrid_forward(self, F, x):
        feat = self.feat(x)
        proj = self.proj(feat)
        out = F.concat(feat, proj, dim=1)
        # if self.use_act:
        #     out = self.act(out)

        return out

class  GhostResBlock(gluon.HybridBlock):
    def __init__(self, out_channels, ksize, stride, padding, expand_channels, use_act=True, in_channels=0, use_bias=True, **kwargs):
        super(GhostResBlock, self).__init__(**kwargs)
        self.stride = stride
        # self.inner_channels = int(in_channels * expand_ratio)
        self.inner_channels = expand_channels
        with self.name_scope():
            self.feat = nn.HybridSequential()
            self.feat.add(GhostModule(self.inner_channels, ksize, 1, padding, in_channels=in_channels, use_bias=use_bias, use_act=False))
            self.feat.add(nn.BatchNorm(in_channels=self.inner_channels))
            self.feat.add(nn.Activation(activation='relu'))
            if self.stride > 1:
                self.feat.add(nn.Conv2D(self.inner_channels, 1, stride, 0, in_channels=self.inner_channels, groups=self.inner_channels))
                self.feat.add(nn.BatchNorm(in_channels=self.inner_channels))
            self.feat.add(GhostModule(out_channels, ksize, 1, padding, in_channels=self.inner_channels, use_bias=use_bias, use_act=False))
            self.feat.add(nn.BatchNorm(in_channels=out_channels))
            self.proj_path = None
            if self.stride > 1 or (out_channels != in_channels):
                self.proj_path = nn.HybridSequential()
                self.proj_path.add(nn.Conv2D(out_channels, 1, stride, 0, in_channels=in_channels))
                self.proj_path.add(nn.BatchNorm(in_channels=out_channels))

    def hybrid_forward(self, F, x):
        out = self.feat(x)
        if self.proj_path is not None:
            x = self.proj_path(x)
        out = F.elemwise_add(out, x)

        return out


class GhostNet(gluon.HybridBlock):
    def __init__(self, cls_nums=1000, out_stages=1, **kwargs):
        super(GhostNet, self).__init__(**kwargs)
        in_channels  = [16, 16, 16, 24, 24, 40,  40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 960, 960, 1280]
        exp_channels = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        stage_n = [2, 2, 2, 6, 5]
        self.out_stages = out_stages

        with self.name_scope():
            ## stem block
            self.stem = BasicConv2D(16, 3, 2, 1, in_channels=3)
            ## stages
            self.stage1 = nn.HybridSequential()
            self.stage1.add(GhostResBlock(16, 3, 1, 1, 16, in_channels=16))
            self.stage1.add(GhostResBlock(24, 3, 2, 1, 48, in_channels=16))
            self.stage2 = nn.HybridSequential()
            self.stage2.add(GhostResBlock(24, 3, 1, 1, 72, in_channels=24))
            self.stage2.add(GhostResBlock(40, 3, 2, 1, 72, in_channels=24))
            self.stage3 = nn.HybridSequential()
            self.stage3.add(GhostResBlock(40, 3, 1, 1, 120, in_channels=40))
            self.stage3.add(GhostResBlock(80, 3, 2, 1, 240, in_channels=40))
            self.stage4 = nn.HybridSequential()
            self.stage4.add(GhostResBlock(80,  3, 1, 1, 200, in_channels=80))
            self.stage4.add(GhostResBlock(80,  3, 1, 1, 184, in_channels=80))
            self.stage4.add(GhostResBlock(80,  3, 1, 1, 184, in_channels=80))
            self.stage4.add(GhostResBlock(112, 3, 1, 1, 480, in_channels=80))
            self.stage4.add(GhostResBlock(112, 3, 1, 1, 672, in_channels=112))
            self.stage4.add(GhostResBlock(160, 3, 2, 1, 672, in_channels=112))
            self.stage5 = nn.HybridSequential()
            self.stage5.add(GhostResBlock(160, 3, 1, 1, 960, in_channels=160))
            self.stage5.add(GhostResBlock(160, 3, 1, 1, 960, in_channels=160))
            self.stage5.add(GhostResBlock(160, 3, 1, 1, 960, in_channels=160))
            self.stage5.add(GhostResBlock(160, 3, 1, 1, 960, in_channels=160))
            self.stage5.add(BasicConv2D(960, 1, 1, 0, in_channels=160))

            ## Head
            self.avg = nn.GlobalAvgPool2D()
            self.fc0 = nn.Dense(1280)
            self.out = nn.Dense(cls_nums)

    def hybrid_forward(self, F, x):
        feat = self.stem(x)
        # print('shape of feat (stem): ', feat.shape)
        feat1 = self.stage1(feat)
        # print('shape of feat (stage 1): ', feat1.shape)
        feat2 = self.stage2(feat1)
        # print('shape of feat (stage 2): ', feat2.shape)
        feat3 = self.stage3(feat2)
        # print('shape of feat (stage 3): ', feat3.shape)
        feat4 = self.stage4(feat3)
        # print('shape of feat (stage 4): ', feat4.shape)
        feat5 = self.stage5(feat4)
        # print('shape of feat (stage 5): ', feat5.shape)        ## in_w // 32, in_h // 32
        avg = self.avg(feat5)
        # print('shape of feat (global avg): ', feat6.shape)
        fc0 = self.fc0(avg)
        # print('shape of feat (fc0): ', feat7.shape)
        out = self.out(fc0)

        if self.out_stages == 1:
            return out
        else:
            outs = [feat1, feat2, feat3, feat4, feat5]
            return outs[int(-1 * self.out_stages):]


def ghostnet512(cls_nums=1000, pretrained=False, **kwargs):
    net = GhostNet(cls_nums=cls_nums)
    if pretrained:
        pass
    return net


if __name__ == '__main__':
    # data = nd.random.uniform(0.0, 1.0, (1, 3, 224, 224))
    data = nd.random.uniform(0.0, 1.0, (1, 3, 512, 512))
    net = GhostNet()
    net.initialize()
    output = net(data)
    print('shape of output: ', output.shape)
