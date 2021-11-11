# coding:utf-8
import unittest

from net.dataset import VOCDataset


class TestVOCDateset(unittest.TestCase):
    """ 测试 VOC 数据集类 """

    def test_get_image_paths(self):
        """ 测试获取图片路径 """
        voc = VOCDataset('data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007', 'test')
        print('\n', voc.image_paths[-1], len(voc.image_paths))
