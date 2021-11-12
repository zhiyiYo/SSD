# coding:utf-8
import unittest
import torch

from net import SSD


class TestSSD(unittest.TestCase):
    """ 测试 SSD 模型 """

    def test_forward(self):
        """ 测试前馈过程 """
        ssd = SSD(21, [0.1, 0.2])
        print(ssd)
        x = torch.rand(2, 3, 300, 300)
        loc, conf, priors = ssd(x)
        self.assertEqual(loc.shape, (2, 8732, 4))
        self.assertEqual(conf.shape, (2, 8732, 21))
        self.assertEqual(priors.shape, (8732, 4))

    def test_predict(self):
        """ 测试预测边界框 """
        ssd = SSD(21, [0.1, 0.2])
        x = torch.rand(2, 3, 300, 300)
        y = ssd.predict(x)
        print('\n', y)
        self.assertEqual(y.shape, (2, 21, 200, 5))