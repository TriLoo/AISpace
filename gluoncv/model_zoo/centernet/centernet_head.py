""" @package centernet_head.py
    
    implementation of Poly-YOLO head to upsample the feature by 4-folder
    @author smh
    @date 2020.06.26
    @copyright
"""

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

from centernet_common import BasicConvBlock

__all__ = ['StairUpsampling', 'HeatmapConvBlock', 'TagConvBlock', 'RegConvBlock', 'TransposeGatherFeats']

class StairUpsamplingUnit(gluon.HybridBlock):
    def __init__(self, out_channels=256, kernel_size=3, stride=1, padding=1, use_bias=True, **kwargs):
        ## here, default use kernel_size = 3, not 1 in the paper
        super(StairUpsamplingUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.bottom = gluon.nn.HybridSequential()
            self.bottom.add(
                nn.Conv2D(out_channels, kernel_size, stride, padding, use_bias=use_bias),
                nn.BatchNorm(),
                nn.Activation(activation='relu')
            )
            self.up = gluon.nn.HybridSequential()
            self.up.add(
                nn.Conv2D(out_channels, kernel_size, stride, padding, use_bias=use_bias),
                nn.BatchNorm(),
                nn.Activation(activation='relu')
            )

    def hybrid_forward(self, F, x, y):
        ## here, first upsampling the bottom input, then conv, not like the paper, which is inverse
        x = F.contrib.BilinearResize2D(x, scale_height=2, scale_width=2, align_corners=True)
        bottom_feat = self.bottom(x)
        up_feat = self.up(y)
        output = F.elemwise_add(bottom_feat, up_feat)

        return output


class StairUpsampling(gluon.HybridBlock):
    def __init__(self, c_out = 256, n_stages=2, **kwargs):
        super(StairUpsampling, self).__init__(**kwargs)
        self.n_stages = n_stages
        with self.name_scope():
            for i in range(n_stages-1):
                stage = gluon.nn.HybridSequential()
                stage.add(StairUpsamplingUnit(c_out))
                setattr(self, 'stage_{}'.format(str(i)), stage)
    
    def hybrid_forward(self, F, x):
        ## x: bottom in, y: up in
        assert len(x) == self.n_stages, "Input num mismatach: fm nums {}, expected stage nums {}".format(str(len(x)), str(self.n_stages))
        out = x[0]
        for i in range(self.n_stages-1):
            stage = getattr(self, 'stage_{}'.format(str(i)))
            out = stage(out, x[i+1])

        return out


class HeatmapConvBlock(gluon.HybridBlock):
    def __init__(self, c_out, c_mid, kernel_size=3, strides=1, padding=1, use_bias=True, in_c=0, **kwargs):
        super(HeatmapConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = BasicConvBlock(c_mid, kernel_size, strides, padding, use_bias, in_c, False)
            self.out = nn.Conv2D(c_out, 1, 1, 0, in_channels=c_mid)
    
    def hybrid_forward(self, F, x):
        feat = self.conv0(x)
        out = self.out(feat)

        return out


class TagConvBlock(gluon.HybridBlock):
    def __init__(self, c_out=1, c_mid=256, kernel_size=3, strides=1, padding=1, use_bias=True, in_c=0, **kwargs):
        super(TagConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = BasicConvBlock(c_mid, kernel_size, strides, padding, use_bias, in_c, False)
            self.out = nn.Conv2D(c_out, 1, 1, 0, in_channels=c_mid)
    
    def hybrid_forward(self, F, x):
        feat = self.conv0(x)
        out = self.out(feat)

        return out

class RegConvBlock(gluon.HybridBlock):
    def __init__(self, c_out=2, c_mid=256, kernel_size=3, strides=1, padding=1, use_bias=True, in_c=0, **kwargs):
        super(RegConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = BasicConvBlock(c_mid, kernel_size, strides, padding, use_bias, in_c, False)
            self.out = nn.Conv2D(c_out, 1, 1, 0, in_channels=c_mid)
    
    def hybrid_forward(self, F, x):
        feat = self.conv0(x)
        out = self.out(feat)

        return out


class GatherFeats(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(GatherFeats, self).__init__(**kwargs)
    
    def hybrid_forward(self, F, feat, ind, mask=None):
        '''
            @param ind: 1-D input index to select the corresponding rows of feat !
            @param mask: 1-D input index too!
        '''
        feat = F.take(feat, ind, axis=1)
        if mask is not None:
            feat = F.take(feat, mask, axis=1)
            feat = F.reshape(feat, shape=(-3, 0))
        return feat

class TransposeGatherFeats(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(TransposeGatherFeats, self).__init__(**kwargs)
        self.gather = GatherFeats()

    def hybrid_forward(self, F, feat, ind):
        feat = F.transpose(feat, axes=(0, 2, 3, 1))
        feat = F.reshape(feat, shape=(0, -3, 0))
        feat = self.gather(feat, ind)

        return feat


if __name__ == '__main__':
    fm0 = nd.random.uniform(0.0, 1.0, (1, 3, 16, 16))
    fm1 = nd.random.uniform(0.0, 1.0, (1, 3, 32, 32))
    fm2 = nd.random.uniform(0.0, 1.0, (1, 3, 64, 64))

    x = [fm0, fm1, fm2]
    net = StairUpsampling(n_stages=3)
    net.initialize()
    net.hybridize()

    output = net(x)
    print('output: ', output.shape)
