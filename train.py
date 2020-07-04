import os
import torch
import numpy as np
from net.resnet import resnet
from data.pascal_voc import PASCAL_VOC
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
import config as cfg

def train():
    np.random.seed(cfg.rng_seed)

    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    train_set = PASCAL_VOC(cfg.trainset_root_path, 'trainval')
    dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    iters_per_epoch = len(train_set) // cfg.batch_size

    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    fasterRCNN = resnet(cfg.backbone, pretrained=True, class_agnostic=True)
    fasterRCNN.create_architecture()

    optimizer = torch.optim.SGD(fasterRCNN.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=5e-4)

    fasterRCNN.to(device)

    im_data = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data = im_data.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    gt_boxes = Variable(gt_boxes)

    start_epoch = 0

    #load from
    if not cfg.load_from is None:
        checkpoint = torch.load(cfg.load_from)
        fasterRCNN.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for ep in range(start_epoch, cfg.epoch):

        fasterRCNN.train()

        if ep != 0 and (ep + 1) % cfg.lr_decay_step == 0:
            cur_lr = get_learing_rate(optimizer)
            adjust_learning_rate(optimizer, cur_lr * cfg.lr_decay)

        for step, batch_data in enumerate(dataloader):
            with torch.no_grad():
                im_data.resize_(batch_data['image'].size()).copy_(batch_data['image'])
                gt_boxes.resize_(batch_data['gt_boxes'].size()).copy_(batch_data['gt_boxes'])

            fasterRCNN.zero_grad()
            print('[epoch:{}/{}], [step {}/{}]'.format(ep + 1, cfg.epoch, step + 1, iters_per_epoch))

            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    roi_labels = fasterRCNN(im_data, gt_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            optimizer.zero_grad()
            loss.backward()
            if cfg.clip_grad:
                clip_grad_norm(fasterRCNN.parameters(), 10)  #限制每个梯度，防止梯度爆炸
            optimizer.step()

            cur_lr = get_learing_rate(optimizer)
            print('loss:{:.5f}, lr:{}, rpn cls loss:{:.5f}, rpn bbox loss:{:.5f}, rcnn cls loss:{:.5f}, rcnn bbox loss:{:.5f}'.format(
                loss.item(),
                cur_lr,
                rpn_loss_cls.item(),
                rpn_loss_bbox.item(),
                RCNN_loss_cls.item(),
                RCNN_loss_bbox.item()
            ))
            print('cls_prob:', cls_prob)

        #一个epoch结束后，则保存模型
        if ep % (cfg.checkpoint_interval + 1) == 0:
            state = {
                'epoch': ep,
                'model_state_dict':fasterRCNN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            save_path = os.path.join(cfg.work_dir, cfg.checkpoint_name + '-' + str(ep+1) + '.pth')
            torch.save(state, save_path)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learing_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    train()