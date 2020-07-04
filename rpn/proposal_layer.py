import torch
import torch.nn as nn
import config as cfg
import numpy as np
from torchvision.ops import nms
from rpn.generate_anchors import generate_anchors
from rpn.bbox_transform import bbox_transform_inv, clip_boxes


class _ProposalLayer(nn.Module):
    def     __init__(self, feature_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feature_stride
        self._anchors = torch.from_numpy(generate_anchors(feature_stride=16,
                                                          scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        # 按照通道C取出RPN预测的框属于前景的分数，请注意，在_num_anchors*2个channel中，
        # 前_num_anchors个是框属于背景的概率，后_num_anchors个才是属于前景的概率
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        image_width = input[2]
        image_height = input[3]
        is_training = input[4]

        if is_training:
            pre_nms_topN = cfg.train_rpn_pre_nms_top_N
            post_nms_topN = cfg.train_rpn_post_nms_top_N
            nms_thresh = cfg.rpn_nms_thresh
        else:
            pre_nms_topN = cfg.test_rpn_post_nms_top_N
            post_nms_topN = cfg.test_rpn_post_nms_top_N
            nms_thresh = cfg.rpn_nms_thresh

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        proposals = clip_boxes(proposals, (image_height, image_width), batch_size)  #将超出范围的候选框给夹紧使其不超过图像范围

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]  #选取最高的前pre_nms_topN个

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i   #属于哪个batch
            output[i, :num_proposal, 1:] = proposals_single  #候选框坐标

        return output


    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))

        return keep


