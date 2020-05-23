# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
import numpy as np


class UnetSkipConnectionBlock(gluon.HybridBlock):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm, use_dropout=False, **kwargs):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm
        if input_nc is None:
            input_nc = outer_nc

        with self.name_scope():
            if outermost:
                downconv = nn.Conv2D(inner_nc, kernel_size=4, strides=2, padding=1, use_bias=use_bias, in_channels=input_nc)
                uprelu = nn.Activation(activation='relu')
                upconv = nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1, in_channels=inner_nc * 2)
                down = [downconv]
                up = [uprelu, upconv, nn.Activation(activation='tanh')]
                model = down + [submodule] + up
            elif innermost:
                downrelu = nn.LeakyReLU(0.2)
                downconv = nn.Conv2D(inner_nc, kernel_size=4, strides=2, padding=1, use_bias=use_bias, in_channels=input_nc)
                upconv = nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1, use_bias=use_bias, in_channels=inner_nc)
                upnorm = norm_layer(in_channels=outer_nc)
                uprelu = nn.Activation(activation='relu')
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down +  up
            else:
                downconv = nn.Conv2D(inner_nc, kernel_size=4, strides=2, padding=1, use_bias=use_bias, in_channels=input_nc)
                downrelu = nn.LeakyReLU(0.2)
                downnorm = norm_layer(in_channels=inner_nc)
                uprelu = nn.Activation(activation='relu')
                upnorm = norm_layer(in_channels=outer_nc)
                upconv = nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1, use_bias=use_bias, in_channels=inner_nc * 2)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

            self.model = nn.HybridSequential()
            self.model.add(*model)
    
    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(x, self.model(x), dim=1)


class UnetGenerator(gluon.HybridBlock):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm, use_dropout=False, **kwargs):
        super(UnetGenerator, self).__init__(**kwargs)

        with self.name_scope():
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
            for _ in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=nn.BatchNorm, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=nn.BatchNorm)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=nn.BatchNorm)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=nn.BatchNorm)

            self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def hybrid_forward(self, F, x):
        return self.model(x)


class Discriminator(nn.HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = nn.HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1)/2))
            self.model.add(nn.Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(nn.LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(nn.Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(nn.BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(nn.LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(nn.Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(nn.BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(nn.LeakyReLU(alpha=0.2))
            self.model.add(nn.Conv2D(channels=1, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult))
            if use_sigmoid:
                self.model.add(nn.Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        #print(out)
        return out


class NLayerDiscriminator(gluon.HybridBlock):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm, **kwargs):
        super(NLayerDiscriminator, self).__init__(**kwargs)
        use_bias = norm_layer == nn.InstanceNorm

        kw = 4
        padw = 1
        nf_mult = 1
        nf_mult_prev = 1

        with self.name_scope():
            sequence = [nn.Conv2D(ndf, kernel_size=kw, strides=2, padding=padw, in_channels=input_nc), nn.LeakyReLU(0.2)]
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [nn.Conv2D(ndf * nf_mult, kernel_size=kw, strides=2, padding=padw, use_bias=use_bias, in_channels=ndf * nf_mult_prev),
                             nn.BatchNorm(in_channels=ndf * nf_mult),
                             nn.LeakyReLU(0.2)]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv2D(ndf * nf_mult, kernel_size=kw, strides=1, padding=padw, use_bias=use_bias, in_channels=ndf * nf_mult_prev),
                norm_layer(in_channels=ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

            sequence += [nn.Conv2D(1, kernel_size=kw, strides=1, padding=padw, in_channels=ndf * nf_mult)]
            self.model = nn.HybridSequential()
            self.model.add(*sequence)


    def hybrid_forward(self, F, x):
        return self.model(x)


class PixelDiscriminator(gluon.HybridBlock):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm, **kwargs):
        super(PixelDiscriminator, self).__init__(**kwargs)
        use_bias = norm_layer == nn.BatchNorm

        net = [
            nn.Conv2D(ndf, kernel_size=1, strides=1, padding=0, in_channels=input_nc),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ndf * 2, kernel_size=1, strides=1, padding=0, use_bias=use_bias),
            norm_layer(in_channels=ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2D(1, kernel_size=1, strides=1, padding=0, use_bias=use_bias, in_channels=ndf * 2)
        ]

        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(*net)

    def hybrid_forward(self, F, x):
        return self.net(x)


if __name__ == '__main__':
    in_data = nd.random.uniform(0.0, 1.0, (1, 3, 256, 256))
    # in_data = nd.random.uniform(0.0, 1.0, (1, 3, 512, 512))
    net = UnetGenerator(3, 3, 8, ngf=64, use_dropout=True)
    net.initialize()
    out_data = net(in_data)
    print('shape of out_data: ', out_data.shape)

    # netD = NLayerDiscriminator(3)
    netD = PixelDiscriminator(3)
    netD.initialize()
    prev_data = netD(out_data)
    print('shape of prev_data: ', prev_data.shape)
