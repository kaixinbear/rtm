# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with no padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)
     

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_FP(nn.Module):
    def _double_up(self, x):
        double_up_x = F.interpolate(x, scale_factor = 2, mode="bilinear")
        return double_up_x
    
    def _deconv_bn_relu(self, in_planes, out_planes, kernel_size, stride, groups=1, padding=1,  output_padding=0, dilation=1):
        conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, output_padding=output_padding, bias=False)
        bn = nn.BatchNorm2d(out_planes)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
        

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        super(ResNet_FP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv_1_1 = conv1x1(512,256)
        self.conv_1_2 = conv1x1(512,256)
        self.conv_2_1 = conv1x1(256,128)
        self.conv_2_2 = conv1x1(256,128)
        self.conv_3_1 = conv1x1(128,64)
        self.conv_3_2 = conv1x1(128,64)
        self.up_to_featmax_1 = self._deconv_bn_relu(256, 64, kernel_size=6, stride=4)
        self.up_to_featmax_2 = self._deconv_bn_relu(128, 64, kernel_size=4, stride=2)
        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=256,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out1 = x
        x = self.layer2(x)
        out2 = x
        x = self.layer3(x)
        out3 = x
        x = self.layer4(x)
        out4 = x
        
        feat_map1 = self.conv_1_2(torch.cat([out3,self.conv_1_1(self._double_up(out4))],dim=1))
        feat_map2 = self.conv_2_2(torch.cat([out2,self.conv_2_1(self._double_up(out3))],dim=1))
        feat_map3 = self.conv_3_2(torch.cat([out1,self.conv_3_1(self._double_up(out2))],dim=1))
        
        resized_feat_map1 = self.up_to_featmax_1(feat_map1)
        resized_feat_map2 = self.up_to_featmax_2(feat_map2)
                 
        b1,c1,h1,w1 = resized_feat_map1.size()
        b2,c2,h2,w2 = resized_feat_map2.size()
        b3,c3,h3,w3 = feat_map3.size()
        feature_map = resized_feat_map1*F.softmax(resized_feat_map1.contiguous().\
                      view(b1,c1,-1),dim=2).view(b1,c1,h1,w1) + \
                        resized_feat_map2*F.softmax(resized_feat_map2.contiguous().\
                      view(b2,c2,-1),dim=2).view(b2,c2,h2,w2) + \
                                feat_map3*F.softmax(feat_map3.contiguous().\
                      view(b3,c3,-1),dim=2).view(b3,c3,h3,w3)
                               
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feature_map)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_FP_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = ResNet_FP(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model
