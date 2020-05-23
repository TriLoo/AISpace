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

## TODO

* CycleGAN: training
* Pix2PixHD
* Deblur V1 & V2
* DPE (dependent on second-order gradient, some op in mxnet not support yet, same as below)
* WGAN
* WGAN-GP
* etc

