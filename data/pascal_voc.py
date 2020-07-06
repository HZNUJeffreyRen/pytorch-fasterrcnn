import torch
import numpy as np
import os
import cv2
import config as cfg
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader


class PASCAL_VOC(Dataset):
    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.phase = phase
        self.class_to_ind = cfg.class_to_ind
        self.data = self.load_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path = self.data[idx]['img_path']
        image = cv2.imread(img_path)
        image, scale = self.prep_im_for_blob(image)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)  #转成nchw的形式
        gt_boxes = self.data[idx]['gt_boxes']
        gt_boxes[:, 0:4] =  gt_boxes[:, 0:4] * scale
        gt_cls = self.data[idx]['gt_classes']
        #gt_boxes = np.concatenate([gt_boxes, gt_cls], axis=1)
        image_height, image_width = image.shape[1:]
        im_info = np.array([image_height, image_width, scale], dtype=float)
        read_data = {'image':torch.from_numpy(image),
                     'im_info':im_info,
                     'gt_classes':torch.from_numpy(gt_cls),
                     'gt_boxes': torch.from_numpy(gt_boxes).float(),
                     'imname': img_path}

        return read_data


    def load_data(self):
        print('loading data......')
        data = list()
        jpeg_images_path = os.path.join(self.root_dir, 'JPEGImages')
        annotations_path = os.path.join(self.root_dir, 'Annotations')
        phase_txt = os.path.join(self.root_dir, 'ImageSets', 'Main', self.phase + '.txt')
        with open(phase_txt, 'r') as fp:
            for img_name in fp:
                data_blob = dict()
                img_path = os.path.join(jpeg_images_path, img_name.strip() + '.jpg')
                xml_path = os.path.join(annotations_path, img_name.strip() + '.xml')

                assert os.path.exists(img_path), img_path + ' is not exists.'
                assert os.path.exists(xml_path), xml_path + ' is not exists.'

                data_blob['img_path'] = img_path
                gt_boxes, gt_classes = self.load_annotations(xml_path)
                data_blob['gt_boxes'] = gt_boxes
                data_blob['gt_classes'] = gt_classes
                data.append(data_blob)
        print('over!')

        return data


    def load_annotations(self, xml_path):
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        # image_size = tree.find('size')
        # size_info = np.zeros((2,), dtype=np.float32)
        # size_info[0] = float(image_size.find('width').text)
        # size_info[1] = float(image_size.find('height').text)
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 5), dtype=np.float32)
        gt_classes = np.zeros((num_objs, 1), dtype=np.int32)
        # difficult = np.empty((num_objs))
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self.class_to_ind[obj.find('name').text.strip()]
            boxes[ix, :] = [x1, y1, x2, y2, cls]
            gt_classes[ix, :] = cls
        # gt_boxes = np.concatenate([boxes, gt_classes], axis=1)

        return boxes, gt_classes


    def prep_im_for_blob(self, im):
        im = im.astype(np.float32, copy=False)
        im -= cfg.pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(cfg.image_min_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.image_max_size:
            im_scale = float(cfg.image_max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)

        return im, im_scale


if __name__ == '__main__':
    train_set = PASCAL_VOC('G:/VOC2007/train/VOC2007/', 'trainval')
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    print(len(train_set))
    for i, data in enumerate(dataloader):
        print(data)
        # print(data['gt_boxes'].shape)
        # if data['gt_boxes'].shape[1] == 0:
        # print(data['image'].shape, data['gt_boxes'].shape, data['gt_classes'].shape)
        # print(data['image'].size())








