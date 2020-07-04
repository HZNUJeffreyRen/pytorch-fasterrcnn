import torch
import config as cfg
import numpy as np
import copy
import cv2
import time
from net.resnet import resnet
from rpn.bbox_transform import bbox_transform_inv, clip_boxes
from torchvision.ops import nms

ind_class = {v : k for k, v in cfg.class_to_ind.items()}

@torch.no_grad()
def inference(_test_img_path, _check_point, _score_threshold=0.3):
    test_img_path = _test_img_path
    check_point = _check_point
    score_threshold = _score_threshold

    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    fasterRCNN = resnet(cfg.backbone, pretrained=False, class_agnostic=True)
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (check_point))
    checkpoint = torch.load(check_point)
    fasterRCNN.load_state_dict(checkpoint['model_state_dict'])
    print('load model successfully!')

    fasterRCNN.eval()
    fasterRCNN.to(device)

    im_data = torch.FloatTensor(1)
    im_data = im_data.cuda()

    start_time = time.time()

    test_img = cv2.imread(test_img_path)

    test_img_copy = copy.deepcopy(test_img)
    test_img_copy = image_preprocess(test_img_copy)
    test_img_copy = torch.from_numpy(test_img_copy)
    im_data.resize_(test_img_copy.shape).copy_(test_img_copy)

    rois, cls_prob, bbox_pred, _, _, _, _, _ = fasterRCNN(im_data, None)  #without gt
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    box_deltas = bbox_pred.data
    if cfg.bbox_normalize_targets_precomputed:
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.bbox_normalize_std).cuda() \
                     + torch.FloatTensor(cfg.bbox_normalize_means).cuda()
        box_deltas = box_deltas.view(1, -1, 4)
    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, (im_data.size(2), im_data.size(3)), 1)

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    for j in range(1, len(cfg.class_to_ind)):
        inds = torch.nonzero(scores[:, j] > score_threshold).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]

            cls_dets = cls_boxes[order]
            cls_scores = cls_scores.squeeze()[order]

            keep = nms(cls_dets, cls_scores, cfg.test_nms_threshold)
            cls_dets = cls_dets[keep.view(-1).long()]  #当前类别保留下来的目标框
            cls_scores = cls_scores[keep.view(-1).long()]
            test_img = draw_target(test_img, cls_dets, cls_scores, j)

    end_time = time.time()
    print('detect time:{}s'.format(end_time-start_time))

    cv2.imshow('result', test_img)
    cv2.waitKey(0)


def image_preprocess(im):
    im = im.astype(np.float)
    im -= cfg.pixel_means
    im = im[np.newaxis, :, :, :]
    im = im.transpose(0, 3, 1, 2) #nchw

    return im


def draw_target(image, dets, scores, class_idx):
    for det, score in zip(dets,scores):
        image = cv2.rectangle(image, (det[0], det[1]), (det[2],det[3]), (0, 0, 255), 1)
        text = '{}  {:.2f}'.format(ind_class[class_idx], score)
        image = cv2.putText(image, text, (det[0], det[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    return image


if __name__ == '__main__':
    test_img_path = './000497.jpg'
    check_point = './work_dir/fasterrcnn_r101-19.pth'
    score_threshold = 0.5
    inference(test_img_path, check_point, score_threshold)