import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import conv1x1, conv3x3, max_pool
import config as cfg

class FPN(nn.Module):

    def __init__(self, feature_inchannels=cfg.neck_feature_inchannels):
        super(FPN, self).__init__()

        self.feature_inchannels = feature_inchannels
        #self.upsample = F.interpolate(scale_factor=2, mode='bilinear') #上采样
        self.conv3x3_1 = conv3x3(feature_inchannels[3], 256)  #3x3卷积用于消除上采样带来的混叠效应
        self.conv3x3_2 = conv3x3(feature_inchannels[2], 256)
        self.conv3x3_3 = conv3x3(feature_inchannels[1], 256)
        self.conv3x3_4 = conv3x3(feature_inchannels[0], 256)
        self.max_pool = max_pool(1, stride=2)
        self.conv1x1_1 = conv1x1(feature_inchannels[2], feature_inchannels[3])
        self.conv1x1_2 = conv1x1(feature_inchannels[1], feature_inchannels[2])
        self.conv1x1_3 = conv1x1(feature_inchannels[0], feature_inchannels[1])

    def forward(self, *input):
        c5, c4, c3, c2 = input
        p6 = self.max_pool(c5)
        p5 = self.conv3x3_1(c5)
        p4 = self.conv3x3_2(F.interpolate(c5, scale_factor=2, mode='bilinear') + self.conv1x1_1(c4))
        p3 = self.conv3x3_3(F.interpolate(c4, scale_factor=2, mode='bilinear') + self.conv1x1_2(c3))
        p2 = self.conv3x3_4(F.interpolate(c3, scale_factor=2, mode='bilinear') + self.conv1x1_3(c2))
        feature_pyramid = [p2, p3, p4, p5, p6]

        return feature_pyramid

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



