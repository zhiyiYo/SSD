# coding:utf-8
import unittest
import torch

from net.dataset import VOCDataset
from torchvision import transforms as T

from utils.augmentation_utils import *
from utils.box_utils import draw


class TestAugmentation(unittest.TestCase):
    """ 测试图像增强 """

    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.dataset = VOCDataset(
            'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007', 'test')

    def test_random_flip(self):
        """ 测试随机翻转 """
        self.dataset.transformer = Compose(
            [BBoxToAbsoluteCoords(), RandomFlip()])
        image, target = self.dataset[4]
        self.draw(image, target)

    def test_random_crop(self):
        """ 测试随机裁剪 """
        self.dataset.transformer = Compose(
            [BBoxToAbsoluteCoords(), RandomSampleCrop()])
        image, target = self.dataset[6]
        self.draw(image, target)

    def test_resize(self):
        self.dataset.transformer = Compose(
            [Resize((500, 500)), BBoxToAbsoluteCoords()])
        image, target = self.dataset[6]
        self.draw(image, target)

    def test_color_jitter(self):
        """ 测试颜色扰动 """
        self.dataset.transformer = Compose(
            [ImageToFloat32(), BBoxToAbsoluteCoords(), ColorJitter()])
        image, target = self.dataset[6]
        self.draw(image, target)

    def test_ssd_augmentation(self):
        self.dataset.transformer = Compose(
            [SSDAugmentation(), BBoxToAbsoluteCoords()])
        image, target = self.dataset[6]
        self.draw(image, target)

    def draw(self, image: torch.Tensor, target):
        """ 绘制图像 """
        image = image.permute(1, 2, 0).numpy()
        label = [self.dataset.classes[int(i)] for i in target[:, 4]]

        # 绘制边界框和标签
        image = draw(image, target[:, :4], label)
        image.show()
