import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rpn.rpn import RPN
from rpn.proposal_target_layer_cascade import ProposalTargetLayer
from utils.net_utils import _smooth_l1_loss
from torchvision.ops import roi_align, roi_pool
import config as cfg


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, class_agnostic, is_training):
        super(_fasterRCNN, self).__init__()

        self.is_training = is_training
        self.n_classes = len(cfg.class_to_ind)
        self.class_agnostic = class_agnostic

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = RPN(self.is_training)
        self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)


    def forward(self, im_data, gt_boxes, im_info):
        batch_size = im_data.size(0)
        im_info = im_info.data


        if not gt_boxes is None:
            gt_boxes = gt_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.pooling_mode == 'align':
            # pooled_feat = self.RCNN_roi_align(feature_map, rois.view(-1, 5))
            pooled_feat = roi_align(base_feat, rois.view(-1, 5), (cfg.pool_size, cfg.pool_size), 1.0/16)
        elif cfg.pooling_mode == 'pool':
            #pooled_feat = self.RCNN_roi_pool(feature_map, rois.view(-1, 5))
            pooled_feat = roi_pool(base_feat, rois.view(-1, 5), (cfg.pool_size, cfg.pool_size), 1.0/16)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.truncated)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.truncated)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.truncated)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.truncated)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.truncated)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()