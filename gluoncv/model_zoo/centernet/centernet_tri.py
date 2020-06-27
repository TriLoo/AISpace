""" @package centernet_tri.py

    @author smh
    @date 2020.06.20
    @copyright
"""

# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from .. import model_zoo

from centernet_common import BasicConvBlock, ResConvBlock
from centernet_head import StairUpsampling, HeatmapConvBlock, TagConvBlock, RegConvBlock, TransposeGatherFeats
from corner_pooling import BottomRightPooling, TopLeftPooling, CrossPooling

class CenterNetTri(gluon.HybridBlock):
    def __init__(self, backbone, n_cls, **kwargs):
        ''' CenterNet

            Backbone: Ghost
            Neck    : Stair Upsampling
            Head    : CenterNet (Cascade Corner Pooling, Center Pooling, Heatmap, Embeding map etc.)
            @params backbone: str or gluon Modules
            @params n_cls: class nums, e.g. 80 for coco dataset, 20 for voc
        '''
        super(CenterNetTri, self).__init__(**kwargs)
        ## backbone
        with self.name_scope():
            ## Stem Block, downscale by 4 total
            # TODO: add stem block here: BasicConvBlock(128, 7, 2, 3, use_bias=False, in_c=3) & ResConvBlock(256, 3, 2, 1, False, 128)
            ## Backbone
            if isinstance(backbone, str):
                self.backbone = model_zoo.get_model(backbone, **kwargs)
            else:
                self.backbone = backbone
            ## Neck
            self.neck = StairUpsampling(cout=256, n_stages=3)
            ## Head
            self.cnvs = BasicConvBlock(256, 3, 1, 1, False, 256, True, True)
            self.tl_cnvs = TopLeftPooling(256)          # cascade top left pooling
            self.br_cnvs = BottomRightPooling(256)      # cascade bottom right pooling
            self.ct_cnvs = CrossPooling(256)            # center pooling
            ## Keypoint heatmaps
            self.tl_heats = HeatmapConvBlock(n_cls, 256, 3, 1, 1, True, 256)
            self.br_heats = HeatmapConvBlock(n_cls, 256, 3, 1, 1, True, 256)
            self.ct_heats = HeatmapConvBlock(n_cls, 256, 3, 1, 1, True, 256)
            ## Tags, same structure as heatmaps except the output channel nums
            self.tl_tags = TagConvBlock(1, 256, 3, 1, 1, True, 256)
            self.br_tags = TagConvBlock(1, 256, 3, 1, 1, True, 256)
            ## Box Regress, same as tag & heatmap output layers, only different at the output channels
            self.tl_regs = RegConvBlock(2, 256, 3, 1, 1, True, 256)
            self.br_regs = RegConvBlock(2, 256, 3, 1, 1, True, 256)
            self.ct_regs = RegConvBlock(2, 256, 3, 1, 1, True, 256)
            ## for training
            self.transpose_gather_feats = TransposeGatherFeats()

    def hybrid_forward(self, F, x, *args):
        if len(args) != 0 and not autograd.is_training():
            raise TypeError('CenterNet inference only need one input data.')

        feats = self.backbone(x)
        feat = self.neck(feats)

        # head convolutions
        tl_cnv = self.tl_cnvs(feat)
        br_cnv = self.br_cnvs(feat)
        ct_cnv = self.ct_cnvs(feat)

        # heatmap
        tl_heat = self.tl_heats(tl_cnv)
        br_heat = self.br_heats(br_cnv)
        ct_heat = self.ct_heats(ct_cnv)
        # tags
        tl_tag = self.tl_tags(tl_cnv)
        br_tag = self.br_tags(br_cnv)
        # regs
        tl_reg = self.tl_regs(tl_cnv)
        br_reg = self.br_regs(br_cnv)
        ct_reg = self.ct_regs(ct_cnv)

        if autograd.is_recording():
            # compose & gather the tag & ind feats
            tl_tag = self.transpose_gather_feats(tl_tag, args[0])
            br_tag = self.transpose_gather_feats(br_tag, args[1])
            tl_reg = self.transpose_gather_feats(tl_reg, args[0])
            br_reg = self.transpose_gather_feats(br_reg, args[1])
            ct_reg = self.transpose_gather_feats(ct_reg, args[2])
            outs = [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_reg, br_reg, ct_reg]
        else:
            outs = [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_reg, br_reg, ct_reg]

        return outs


if __name__ == '__main__':
    data = mx.nd.random.uniform(0.0, 1.0, (1, 3, 512, 512))
    net = CenterNetTri('ghost', 80)
    net.initialize()

    outputs = net(data)
    print('len of outputs: ', len(outputs))
