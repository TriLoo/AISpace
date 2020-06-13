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

## WS-DAN (2020.05, 31)

  <Weakly Supervised Data Augmentation Network> for FGVC (Fine-Grained Visual Classification)

  use the atention map to calculate a crop * drop mask, apply these mask to the original input and generate another two input data, finally, use this three inputs' loss to backward.

  some pytorch op is not support by mxnet, so some used mxnet ops are not efficient as expected!

  for example, when generate the crop mask, I have copied the data from gpu to cpu to calculate the max nonzero areas, this is so slow and I'm going to fix this soon.
  (no `asnumpy()` is used but the code is still so ugly ... (2020.05.31)).

  update (2020.06.06)

  * change the center loss to HybridBlock based implementation, the new implmented method have no need to support symbol operators, such that all convenient ops from nd can be used.
    cf: [mxnet-center-loss - github](https://github.com/ShownX/mxnet-center-loss/blob/master/center_loss.py)
    caution:
      the update of center feature matrix can be a step of backward from another `center_trainer`, which means, the L2Loss's gradient is exactly the update calculation according to the paper.
      An easy deduction of math can verify this.
  * changed the implementation of `drop` & `crop` mask creation
  * now the speed is also accelerated
  * TODO:
    accelerate the calculation of Bilinear Attention Pooling using FFT & IFFT

## Ghost CenterNet (2020.06.13)

  <CenterNet: Keypoint Triplets for Object Detection>

  <GhostNet: More Features from Cheap Operations>

  trained using COCO 2017 train/val split.

 ### Usage

* first copy the \*\_pool.\* file into the `src/operator/contrib/` folder, corresponding python test file can be found in the `unittests` folder
* re-build and re-install the mxnet from source code
* then normal model training & infering scheme

* More details about the implementation of bottom-up pooling etc can be found in the `notes` folder

The training code in under develop ...

## TODO

* CycleGAN: training
* Pix2PixHD
* Deblur V1 & V2
* DPE (dependent on second-order gradient, some op in mxnet not support yet, same as below)
* WGAN
* WGAN-GP
* etc

