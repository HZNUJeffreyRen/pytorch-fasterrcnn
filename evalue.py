import os
import torch
import config as cfg
import numpy as np
import pickle
from tqdm import tqdm
from net.resnet import resnet
from torch.utils.data import DataLoader
from rpn.bbox_transform import bbox_transform_inv,clip_boxes
from data.pascal_voc import PASCAL_VOC
from torchvision.ops import nms
import xml.etree.ElementTree as ET

@torch.no_grad()
def evalue(check_point, cache_path='./result.pkl', class_agnostic=False, ovthresh=0.5, use_07_metric=False):

    ind_class = {v: k for k, v in cfg.class_to_ind.items()}
    class_result_dic = {k: [] for k in cfg.class_to_ind.keys()}  # store every class result

    imagenames = []

    if not os.path.exists(cache_path):

        test_set = PASCAL_VOC(cfg.testset_root_path, 'test')
        dataloader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

        device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        fasterRCNN = resnet(cfg.backbone, is_training=False, pretrained=False, class_agnostic=class_agnostic)
        fasterRCNN.create_architecture()

        print("load checkpoint %s" % (check_point))

        checkpoint = torch.load(check_point)
        fasterRCNN.load_state_dict(checkpoint['model_state_dict'])

        print('load model successfully!')

        fasterRCNN.eval()
        fasterRCNN.to(device)

        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        gt_boxes = torch.FloatTensor(1)
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        gt_boxes = gt_boxes.cuda()

        #detect for result
        for batch_data in tqdm(dataloader):
            # batch_data = dataloader.next()
            with torch.no_grad():
                im_data.resize_(batch_data['image'].size()).copy_(batch_data['image'])
                gt_boxes.resize_(batch_data['gt_boxes'].size()).copy_(batch_data['gt_boxes'])
                im_info.resize_(batch_data['im_info'].size()).copy_(batch_data['im_info'])

                image_name = os.path.basename(batch_data['imname'][0]).split('.')[0]
                imagenames.append(image_name)

                rois, cls_prob, bbox_pred, _, _, _, _, _ = fasterRCNN(im_data, gt_boxes)

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                box_deltas = bbox_pred.data

                if cfg.bbox_normalize_targets_precomputed:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.bbox_normalize_std).cuda() \
                                 + torch.FloatTensor(cfg.bbox_normalize_means).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info, 1)
                pred_boxes = pred_boxes / batch_data['im_info'][0, 2]

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()

                for j in range(1, len(cfg.class_to_ind)):
                    inds = torch.nonzero(scores[:, j] > 0).view(-1)
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)

                        if class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = pred_boxes[order]
                        cls_scores = cls_scores[order]

                        keep = nms(cls_dets, cls_scores, cfg.test_nms_threshold)
                        cls_dets = cls_dets[keep.view(-1).long()]  # 当前类别保留下来的目标框
                        cls_scores = cls_scores[keep.view(-1).long()]

                        for score, bbox in zip(cls_scores, cls_dets):
                            class_result_dic[ind_class[j]].append({
                                    'image_name': image_name,
                                    'score': score,
                                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]
                                })

        print('writting result cache ......')
        with open(cache_path, 'wb') as fp:
            pickle.dump(class_result_dic, fp)
    else:
        with open(os.path.join(cfg.testset_root_path, 'ImageSets', 'Main', 'test.txt')) as fp:
            for line in fp:
                imagenames.append(line.strip())
        with open(cache_path, 'rb') as fp:
            class_result_dic = pickle.load(fp)


    print('computer mAP... ')
    # computer map
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(os.path.join(cfg.testset_root_path, 'Annotations', imagename + '.xml'))

    # extract gt objects for this class
    mAP = 0
    for classname in cfg.class_to_ind.keys():
        if classname == 'BG':
            continue
        print(classname, end=' ')
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        class_result = class_result_dic[classname]
        image_ids = [r['image_name'] for r in class_result]
        confidence = np.array([float(r['score']) for r in class_result])
        BB = np.array([r['bbox'] for r in class_result])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print(ap)
        mAP += ap
    mAP = mAP / (len(cfg.class_to_ind) - 1)

    print('mAP:', mAP)


def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



if __name__ == '__main__':
    check_point = './work_dir/fasterrcnn_r101-19.pth'
    evalue(check_point)





