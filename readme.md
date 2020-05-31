---
title: Some DL Models
---

## Dependency

mxnet == 2.0.0 (commit: de5105824)

gluoncv == 0.7.0

## Pix2Pix

  gluoncv/model\_zoo/gan/pix2pix.py

## YOLOv3 + GIoU (2020.05.23)

  I changed the box loss (MSE) from original paper to the GIoU.

  According to YOLOv4, the Mish action & GIoU Loss can boost the detection mAP greatly, the replace of LeakyReLU to Mish is easy, so only GIoU loss is implemented here.

## WS-DAN

  <Weakly Supervised Data Augmentation Network> for FGVC (Fine-Grained Visual Classification)

  use the atention map to calculate a crop * drop mask, apply these mask to the original input and generate another two input data, finally, use this three inputs' loss to backward.

  some pytorch op is not support by mxnet, so some used mxnet ops are not efficient as expected!

  for example, when generate the crop mask, I have copied the data from gpu to cpu to calculate the max nonzero areas, this is so slow and I'm going to fix this soon.

## TODO

* CycleGAN: training
* Pix2PixHD
* Deblur V1 & V2
* DPE (dependent on second-order gradient, some op in mxnet not support yet, same as below)
* WGAN
* WGAN-GP
* etc

