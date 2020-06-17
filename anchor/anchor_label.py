import numpy as np
import torch
import config as cfg


def calculate_IOU (target_boxes, gt_boxes):  #gt_boxes[num_obj,4] targer_boxes[w*h*k,4]
    num_gt = gt_boxes.shape[0]
    num_tr = target_boxes.shape[0]
    IOU_s = np.zeros((num_gt,num_tr), dtype=np.float)
    for ix in range(num_gt):
        gt_area = (gt_boxes[ix,2]-gt_boxes[ix,0]) * (gt_boxes[ix,3]-gt_boxes[ix,1])
        for iy in range(num_tr):
            iw = min(gt_boxes[ix,2],target_boxes[iy,2]) - max(gt_boxes[ix,0],target_boxes[iy,0])
            if iw > 0:
                ih = min(gt_boxes[ix,3],target_boxes[iy,3]) - max(gt_boxes[ix,1],target_boxes[iy,1])
                if ih > 0:
                    tar_area = (target_boxes[iy,2]-target_boxes[iy,0]) * (target_boxes[iy,3]-target_boxes[iy,1])
                    i_area = iw * ih
                    iou = i_area/float((gt_area+tar_area-i_area))
                    IOU_s[ix,iy] = iou
    IOU_s = np.transpose(IOU_s)
    return IOU_s


def labels_generate (gt_boxes, target_boxes, overlaps_pos, overlaps_neg, im_width, im_height):
    total_targets = target_boxes.shape[0]
    targets_inside = np.where((target_boxes[:,0]>0)&\
                              (target_boxes[:,2]<im_width)&\
                              (target_boxes[:,1]>0)&\
                              (target_boxes[:,3]<im_height))[0]
    targets = target_boxes[targets_inside]
    labels = np.empty((targets.shape[0],), dtype=np.float32)
    labels.fill(-1)
    IOUs = calculate_IOU(targets, gt_boxes)  #计算所有处于图像内部的anchor和所有gt的交并比
    max_gt_arg = np.argmax(IOUs, axis=1)
    max_IOUS = IOUs[np.arange(len(targets_inside)), max_gt_arg]
    labels[max_IOUS < overlaps_neg] = 0
    max_anchor_arg = np.argmax(IOUs, axis=0)
    labels[max_anchor_arg] = 1
    labels[max_IOUS > overlaps_pos] = 1
    anchor_obj = max_gt_arg  #每个anchor 分配的回归目标，gt的索引
    labels = fill_label(labels, total_targets, targets_inside)
    anchor_obj = fill_label(anchor_obj, total_targets, targets_inside, fill=-1)
    anchor_obj = anchor_obj.astype(np.int64)

    return labels, anchor_obj


def fill_label(labels, total_target, target_inside, fill=-1):
    new_labels = np.empty((total_target, ), dtype=np.float32)
    new_labels.fill(fill)
    new_labels[target_inside] = labels
    return new_labels


def anchor_labels_process(boxes, conners, im_width, im_height, anchor_batch=cfg.anchor_batch_size,
                          overslaps_max=cfg.rpn_pos_iou_thr, overslaps_min=cfg.rpn_neg_iou_thr):
    labels, anchor_obj = labels_generate(boxes, conners, overslaps_max, overslaps_min, im_width, im_height)
    labels = labels_filt(labels, anchor_batch)
    return labels, anchor_obj


def labels_filt (labels, anchor_batch):
    """ label filt: get 256 anchor where 50% for positive anchor, 50 for negative anchor"""
    max_fg_num = anchor_batch * 0.5
    fg_inds = np.where(labels==1)[0]
    if len(fg_inds) > max_fg_num:
        disable_inds = np.random.choice(fg_inds, size=int(len(fg_inds) - max_fg_num), replace=False)
        labels[disable_inds] = -1
    max_bg_num = anchor_batch - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > max_bg_num:
        disable_inds = np.random.choice(bg_inds, size=int(len(bg_inds) - max_bg_num), replace=False)
        labels[disable_inds] = -1
    return labels