import torch
import torch.nn as nn
from rpn.rpn import RPN
from neck.fpn import FPN
from backbone.resnet import resnet50
import config as cfg

class Model(nn.Module):
    def __init__(self, backbone, neck, is_training):
        super(Model, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.is_training = is_training

        #每个特征图一个RPN
        self.rpn_s2 = RPN(stride=cfg.anchor_strides[0])
        self.rpn_s3 = RPN(stride=cfg.anchor_strides[1])
        self.rpn_s4 = RPN(stride=cfg.anchor_strides[2])
        self.rpn_s5 = RPN(stride=cfg.anchor_strides[3])
        self.rpn_s6 = RPN(stride=cfg.anchor_strides[4])

        #权重初始化
        self.backbone.init_weights()
        self.neck.init_weights()
        self.rpn_s2.init_weights()
        self.rpn_s3.init_weights()
        self.rpn_s4.init_weights()
        self.rpn_s5.init_weights()
        self.rpn_s6.init_weights()


    def forward(self, *input):
        feature_set = self.backbone(input)
        feature_pyramid = self.neck(feature_set)
        p2, p3, p4, p5, p6 = feature_pyramid
        cls_s2, reg_s2 = self.rpn_s2(p2)
        cls_s3, reg_s3 = self.rpn_s3(p3)
        cls_s4, reg_s4 = self.rpn_s4(p4)
        cls_s5, reg_s5 = self.rpn_s5(p5)
        cls_s6, reg_s6 = self.rpn_s6(p6)
        rpn_cls_outs = [cls_s2, cls_s3, cls_s4, cls_s5, cls_s6]
        rpn_reg_outs = [reg_s2, reg_s3, reg_s4, reg_s5, reg_s6]

        return rpn_cls_outs, rpn_reg_outs


    def build_loss(self, rpn_cls_outs, rpn_reg_outs, gt_boxes):
        s2_rpn_cls_out, s3_rpn_cls_out, \
        s4_rpn_cls_out, s5_rpn_cls_out, s6_rpn_cls_out = rpn_cls_outs
        s2_rpn_reg_out, s3_rpn_reg_out, \
        s4_rpn_reg_out, s5_rpn_reg_out, s6_rpn_reg_out = rpn_reg_outs

        s2_w = s2_rpn_cls_out.size()[2]
        s2_h = s2_rpn_cls_out.size()[3]
        s2_loss = self.rpn_s2().build_loss(s2_rpn_cls_out, s2_rpn_reg_out, gt_boxes, s2_w, s2_h)

        s3_w = s3_rpn_cls_out.size()[2]
        s3_h = s3_rpn_cls_out.size()[3]
        s3_loss = self.rpn_s3().build_loss(s3_rpn_cls_out, s3_rpn_reg_out, gt_boxes, s3_w, s3_h)

        s4_w = s4_rpn_cls_out.size()[2]
        s4_h = s4_rpn_cls_out.size()[3]
        s4_loss = self.rpn_s4().build_loss(s4_rpn_cls_out, s4_rpn_reg_out, gt_boxes, s4_w, s4_h)

        s5_w = s5_rpn_cls_out.size()[2]
        s5_h = s5_rpn_cls_out.size()[3]
        s5_loss = self.rpn_s5().build_loss(s5_rpn_cls_out, s5_rpn_reg_out, gt_boxes, s5_w, s5_h)

        s6_w = s6_rpn_cls_out.size()[2]
        s6_h = s6_rpn_cls_out.size()[3]
        s6_loss = self.rpn_s6().build_loss(s6_rpn_cls_out, s6_rpn_reg_out, gt_boxes, s6_w, s6_h)

        loss = s2_loss + s3_loss + s4_loss + s5_loss + s6_loss

        return loss

if __name__ == '__main__':
    FasterRCNN = Model(resnet50(pretrained=True),
                       FPN(),
                       is_training=True)

