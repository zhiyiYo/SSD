# coding:utf-8
import unittest

import torch
from net import SSDLoss, SSD


class TestLoss(unittest.TestCase):
    """ 测试损失函数 """

    def test_loss(self):
        # 模型
        model = SSD(21).cuda()
        x = torch.rand((2, 3, 300, 300)).cuda()

        # 预测值
        loc_pred, conf_pred, priors = model(x)

        # 标签
        bbox = torch.randn(2, 10, 4).abs()
        conf = torch.randint(20, (2, 10, 1))
        target = torch.cat((bbox, conf.float()), dim=2)

        # 损失函数
        loss = SSDLoss(21)
        loc_loss, class_loss = loss((loc_pred, conf_pred, priors), target)
        print('\n\tloc_loss = ', loc_loss)
        print('\tclass_loss = ', class_loss)
