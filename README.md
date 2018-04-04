# MobileNet-MXNet

### Introduction

This is a MXNet implementation of Google's MobileNets. For details, please read the original paper:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)


### Pretrained Models on ImageNet

A pretrained MobileNet model on ImageNet is provided and you can use score.py to reproduce the accuracy in ImageNet 2012 val dataset.

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

Network|Top-1|Top-5|model size
:---:|:---:|:---:|---:|
MobileNet| 71.24| 90.15| 16.6MB |
MobileNet-V2 | 71.62 | 90.27| 13.8MB |


### Notes

- RGB mean values **[123.68,116.78,103.94]** are subtracted
- **scale: 0.017** is used as std values for image preprocessing
- This model is converted from [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
- MXNet 11.0rc supports depthwiseConvolution now!
