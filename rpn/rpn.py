import torch
import torch.nn as nn
from layer import conv1x1, conv3x3
import config as cfg

class RPN(nn.Module):

    def __init__(self, inchannels=cfg.rpn_inchannels, feature_channels=cfg.rpn_featurechannels,
                 anchor_scales=cfg.anchor_scales, anchor_ratios=cfg.anchor_ratios):
        super(RPN, self).__init__()
        self.inchannels = inchannels
        self.feature_channels = feature_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

        out_chs = len(self.anchor_scales) * len(self.anchor_ratios)

        self.rpn_conv = conv3x3(self.inchannels, self.feature_channels)
        self.rpn_cls = conv3x3(self.feature_channels, out_chs)
        self.rpn_reg = conv3x3(self.feature_channels, out_chs * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
       feature = self.rpn_conv(input)
       feature = self.relu(feature)

       rpn_cls_score = self.rpn_cls(feature)
       rpn_bbox_pred = self.rpn_reg(feature)

       return rpn_cls_score, rpn_bbox_pred







