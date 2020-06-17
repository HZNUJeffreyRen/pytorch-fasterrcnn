import torch
import torch.nn as nn
import numpy as np
from layer import conv3x3
from anchor.anchor_gennerate import gennerate_all_anchors
from anchor.anchor_label import anchor_labels_process
import config as cfg


class RPN(nn.Module):

    def __init__(self, stride, inchannels=cfg.rpn_inchannels, feature_channels=cfg.rpn_featurechannels,
                 anchor_scales=cfg.anchor_scales, anchor_ratios=cfg.anchor_ratios):
        super(RPN, self).__init__()
        self.inchannels = inchannels
        self.feature_channels = feature_channels
        self.stride = stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)

        self.rpn_conv = conv3x3(self.inchannels, self.feature_channels)
        self.rpn_cls = conv3x3(self.feature_channels, self.anchor_num * 2)
        self.rpn_reg = conv3x3(self.feature_channels, self.anchor_num * 4)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, input):
       feature = self.rpn_conv(input)
       feature = self.relu(feature)

       rpn_cls_score = self.rpn_cls(feature)
       rpn_cls_score = self.relu(rpn_cls_score)
       rpn_bbox_pred = self.rpn_reg(feature)
       rpn_bbox_pred = self.relu(rpn_bbox_pred)

       return rpn_cls_score, rpn_bbox_pred


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def build_loss(self, rpn_cls_score, rpn_bbox_pred, gt_boxes, image_width, image_height, sigma=3.0):

        all_anchors = gennerate_all_anchors(image_width, image_height, self.stride)
        anchor_labels, anchor_objs = anchor_labels_process(gt_boxes, all_anchors, image_width, image_height)
        useful_index = np.where(anchor_labels > 0)  #标签为1的锚点框是要计算回归损失的
        useful_anchor_obj = anchor_objs[useful_index]

        rpn_cls_score_resh = torch.squeeze(rpn_cls_score)
        rpn_cls_score_resh = torch.reshape(rpn_cls_score_resh, [-1, 2 * self.anchor_num])
        rpn_cls_score_resh = torch.reshape(rpn_cls_score_resh, [-1, 2])
        rpn_bbox_pred_resh = torch.squeeze(rpn_bbox_pred)
        rpn_bbox_pred_resh = torch.reshape(rpn_bbox_pred_resh, [-1, 4 * self.anchor_num])
        rpn_bbox_pred_resh = torch.reshape(rpn_bbox_pred_resh, [-1, 4])

        #cls loss
        anchor_labels = torch.from_numpy(anchor_labels).cuda()
        anchor_labels = anchor_labels.int()
        rpn_cls_loss = nn.CrossEntropyLoss(rpn_cls_score_resh, anchor_labels, ignore_index=-1)  #标签为-1的anchor不计入分类损失中

        #回归损失
        fg_anchors = all_anchors[useful_index]  #筛选出前景的锚点框
        target_bbox = gt_boxes[useful_anchor_obj]     #得到每个锚点框的
        fg_anchors = torch.from_numpy(fg_anchors).cuda()
        fg_anchors = fg_anchors.float()
        target_bbox = torch.from_numpy(target_bbox).cuda()
        target_bbox = target_bbox.float()

        #计算回归的目标
        anchor_x1 = fg_anchors[:, 0]
        anchor_y1 = fg_anchors[:, 1]
        anchor_x2 = fg_anchors[:, 2]
        anchor_y2 = fg_anchors[:, 3]

        #转换成ccwh的形式
        re_anchor0 = (anchor_x1 + anchor_x2) / 2.0  #中心点
        re_anchor1 = (anchor_y1 + anchor_y2) / 2.0
        re_anchor2 = anchor_x2 - anchor_x1  #w\h
        re_anchor3 = anchor_y2 - anchor_y1

        target_bbox_x1 = target_bbox[:, 0]
        target_bbox_y1 = target_bbox[:, 1]
        target_bbox_x2 = target_bbox[:, 2]
        target_bbox_y2 = target_bbox[:, 3]

        re_target0 = (target_bbox_x1 + target_bbox_x2) / 2.0
        re_target1 = (target_bbox_y1 + target_bbox_y2) / 2.0
        re_target2 = target_bbox_x2 - target_bbox_x1
        re_target3 = target_bbox_y2 - target_bbox_y1

        bbox_ground_truth_0 = (re_target0 - re_anchor0) / re_anchor2
        bbox_ground_truth_1 = (re_target1 - re_anchor1) / re_anchor3
        bbox_ground_truth_2 = torch.log(re_target2 / re_anchor2)
        bbox_ground_truth_3 = torch.log(re_target3 / re_anchor3)
        bbox_ground_truth = torch.stack((bbox_ground_truth_0,
                                         bbox_ground_truth_1,
                                         bbox_ground_truth_2,
                                         bbox_ground_truth_3), dim=1)

        useful_anchor_obj = torch.from_numpy(useful_anchor_obj).cuda()
        useful_anchor_obj = useful_anchor_obj.detach()
        rpn_bbox_pred_resh_slect = torch.index_select(rpn_bbox_pred_resh, 0, useful_anchor_obj)

        sigma_2 = sigma ** 2
        box_diff = rpn_bbox_pred_resh_slect - bbox_ground_truth
        abs_box_diff = torch.abs(box_diff)
        smoothL1_sign = (abs_box_diff < 1. / sigma_2).detach().float()
        rpn_reg_loss = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                   + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        for i in sorted([0, 1], reverse=True):
            loss_box = loss_box.sum(i)
        rpn_reg_loss = loss_box.mean()

        return rpn_cls_loss, rpn_reg_loss


if __name__ == '__main__':
    # s1 = RPN()
    # s1.build_loss(None,
    #               None,
    #               np.array([[0, 40, 20, 100], [40, 100, 160, 188]]),
    #               400,
    #               400,
    #               cfg.anchor_strides[0]
    #               )
    data = np.array([1, 2, 3, 4])
    data2 = np.array([1, 3, 5, 7])
    data = torch.from_numpy(data).cuda()
    data2 = torch.from_numpy(data2).cuda()
    out = torch.stack((data, data2), dim=1)
    print(out.size())










