# coding:utf-8
import unittest

import torch
from net import SSDLoss, SSD, VOCDataset
from utils.augmentation_utils import Resize


class TestLoss(unittest.TestCase):
    """ 测试损失函数 """

    def test_loss(self):
        # 模型
        model = SSD(21).cuda()
        model.load('model/SSD.pth')

        root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
        dataset = VOCDataset(root, 'trainval', Resize())

        image, target = dataset[0]
        image = image.unsqueeze(0).cuda()
        target = [torch.Tensor(target)]

        # 预测值
        loc_pred, conf_pred, priors = model(image)

        # 损失函数
        loss = SSDLoss(21)
        loc_loss, class_loss = loss((loc_pred, conf_pred, priors), target)
        print('\n\tloc_loss = ', loc_loss)
        print('\tclass_loss = ', class_loss)
