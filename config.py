neck = 'FPN'
neck_feature_inchannels = [256, 512, 1024, 2048]

rpn_inchannels = 256
rpn_featurechannels = 256

anchor_scales = [8]
anchor_ratios = [0.5, 1.0, 2.0]
anchor_strides=[4, 8, 16, 32, 64]

anchor_batch_size = 256
rpn_pos_iou_thr = 0.7
rpn_neg_iou_thr = 0.3

