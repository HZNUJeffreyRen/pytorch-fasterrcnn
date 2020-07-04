import numpy as np

class_to_ind = {
    'BG':0,           'aeroplane':1,       'bicycle':2,    'bird':3,
    'boat':4,    'bottle':5, 'bus':6,    'car':7,
    'cat':8, 'chair':9,     'cow':10,       'diningtable':11,
    'dog':12,   'horse':13,     'motorbike':14,    'pottedplant':15,
    'sheep':16,'sofa':17,   'train':18,    'tvmonitor':19,
    'person':20
}

trainset_root_path = 'G:/VOC2007/train/VOC2007/'
rng_seed = 3
batch_size = 1  #目前只支持batch_size = 1!
epoch = 20
learning_rate = 0.001
lr_decay_step = 5  #epoch
lr_decay = 0.1
checkpoint_interval = 1
checkpoint_name = 'fasterrcnn_r101'
pre_train = True
clip_grad = True
work_dir = './work_dir'
load_from = None
truncated = False

image_max_size = 1000
image_min_size = 600
pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])   #BGR order

backbone='resnet101'
rpn_featurechannels = 512

anchor_scales = [8, 16, 32]
anchor_ratios = [0.5, 1.0, 2.0]

anchor_batch_size = 256
rpn_pos_iou_thr = 0.7
rpn_neg_iou_thr = 0.3

rpn_nms_thresh = 0.7

train_rpn_pre_nms_top_N = 12000
train_rpn_post_nms_top_N = 2000
test_rpn_pre_nms_top_N = 6000
test_rpn_post_nms_top_N = 300

rpn_bbox_inside_weight = (1.0, 1.0, 1.0, 1.0)

proposal_batch_size = 128
proposal_fg_fraction = 0.25
proposal_fg_thresh = 0.5
proposal_bg_thresh_hi = 0.5
proposal_bg_thresh_lo = 0.1

bbox_normalize_targets_precomputed = True
bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
bbox_normalize_std = (0.1, 0.1, 0.2, 0.2)
bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)

pooling_mode = 'align'
pool_size = 7

rcnn_feat_channels = 2048

test_nms_threshold = 0.5


