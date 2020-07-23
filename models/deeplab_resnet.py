import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
affine_par = True
from pretrainedmodels import inceptionresnetv2
import functools
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class backbone(nn.Module):

    def __init__(self, num_filters=256):

        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        )  # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )  # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        )  # 2080

        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # enc = []
        lateral = []
        # enc.append(enc4)
        # enc.append(enc3)
        # enc.append(enc2)
        # enc.append(enc1)
        # enc.append(enc0)

        lateral.append(lateral4)
        lateral.append(lateral3)
        lateral.append(lateral2)
        lateral.append(lateral1)
        lateral.append(lateral0)


        infos = []
        for k in range(1,4):
            infos.append(F.interpolate(lateral[0], lateral[k].size()[2:], mode='bilinear', align_corners=True))
        #print("######################################### ")
        #print("infos[0] ", infos[0].size())
        #print("infos[1] ", infos[1].size())
        #print("infos[2] ", infos[2].size())

        return  lateral , infos




def semantic():
    model = backbone()
    return model
