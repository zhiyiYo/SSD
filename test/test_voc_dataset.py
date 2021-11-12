# coding:utf-8
import unittest

from net.dataset import VOCDataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from utils.augmentation_utils import SSDAugmentation


class TestVOCDateset(unittest.TestCase):
    """ 测试 VOC 数据集类 """

    def __init__(self, methodName) -> None:
        super().__init__(methodName=methodName)
        self.dataset = VOCDataset(
            'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007', 'test')

    def test_get_image_paths(self):
        """ 测试获取图片路径 """
        print('\n', self.dataset.image_paths[-1],
              len(self.dataset.image_paths))

    def test_get_item(self):
        """ 测试获取数据 """
        image, target = self.dataset[-1]
        image = ToPILImage()(image)
        image.show()

    def test_data_loader(self):
        """ 测试数据加载 """
        def collate_fn(batch):
            images = []
            targets = []

            for img, target in batch:
                images.append(img.to(torch.float32))
                targets.append(torch.Tensor(target))

            return torch.stack(images, 0), targets

        self.dataset.transformer = SSDAugmentation()
        data_loader = DataLoader(
            self.dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
        data_iter = iter(data_loader)
        data = next(data_iter)
