import numpy as np
import config as cfg


def gennerate_all_anchors(feature_width, feature_height, anchor_stride,
                          anchor_scales=cfg.anchor_scales, anchor_ratios=cfg.anchor_ratios):
    anchor_base = generate_base_anchors(anchor_stride, anchor_scales, anchor_ratios)

    x_offs = np.arange(0, feature_width - anchor_stride / 2, anchor_stride)
    y_offs = np.arange(0, feature_height- anchor_stride / 2, anchor_stride)

    all_anchors = []
    for x_off in x_offs:
        for y_off in y_offs:
            a = anchor_base.copy()  #这里要用深拷贝，在numpy中，为了提高效率，这种切片出来的子矩阵其实都是原矩阵的引用而已，所以改变子矩阵，原矩阵还是会变的
            a[:, 0] += x_off
            a[:, 1] += y_off
            all_anchors.append(a)
    all_anchors = np.vstack(all_anchors)
    all_anchors = coord_transform(all_anchors)

    return all_anchors


#特征图上第一个点对应的anchor
def generate_base_anchors(anchor_stride, anchor_scales, anchor_ratios):
    anchor_base_x = anchor_stride / 2
    anchor_base_y = anchor_stride / 2
    number_anchors = len(anchor_scales) * len(anchor_ratios)

    anchor_scales = np.array(anchor_scales)
    anchor_ratios = np.array(anchor_ratios)
    anchors_size = get_size_anchors(anchor_scales, anchor_ratios)

    anchor_base = np.zeros((number_anchors, 4), dtype=np.float32)
    anchor_base[:, 0:-1] = anchor_base_x
    anchor_base[:, 1:-1] = anchor_base_y
    anchor_base[:, 2] = anchors_size[:, 0] * anchor_stride
    anchor_base[:, 3] = anchors_size[:, 1] * anchor_stride

    return anchor_base  #对应原始图像的anchor大小


def get_size_anchors(anchor_scales, anchor_ratios):
    anchors_w = []
    anchors_h = []
    for ratio in anchor_ratios:
        w = np.round(np.sqrt(anchor_scales * anchor_scales / ratio))
        h = np.round(np.sqrt(anchor_scales * anchor_scales * ratio))
        anchors_w.append(w)
        anchors_h.append(h)
    anchors_w = np.array(anchors_w).reshape(-1, 1)
    anchors_h = np.array(anchors_h).reshape(-1, 1)

    return np.hstack([anchors_w, anchors_h])


def coord_transform(anchors):
    w = anchors[:, 2]
    h = anchors[:, 3]
    xmin = np.round(anchors[:, 0] - 0.5 * w)
    ymin = np.round(anchors[:, 1] - 0.5 * h)
    xmax = np.round(anchors[:, 0] + 0.5 * w)
    ymax = np.round(anchors[:, 1] + 0.5 * h)

    return np.stack([xmin, ymin, xmax, ymax], axis=1)


if __name__ == '__main__':
    anchors = gennerate_all_anchors(400, 400, anchor_stride=cfg.anchor_strides[0])
    print(anchors)




