# coding:utf-8
from typing import List, Tuple
from os import path
from typing import Dict, List, Union
from xml.etree import ElementTree as ET

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentation_utils import SSDAugmentation, Transformer


class AnnotationTransformer:
    """ xml 格式的标注转换器 """

    def __init__(self, class_to_index: Dict[str, int], keep_difficult=False):
        """
        Parameters
        ----------
        class_to_index: Dict[str, int]
            类别:编码 字典

        keep_difficulty: bool
            是否保留 difficult 为 1 的样本
        """
        self.class_to_index = class_to_index
        self.keep_difficult = keep_difficult

    def __call__(self, file_path: str):
        """ 解析 xml 标签文件

        Parameters
        ----------
        file_path: str
            文件路径

        Returns
        -------
        target: List[list] of shape `(n_objects, 5)`
            标签列表，每个标签的前四个元素为归一化后的边界框，最后一个标签为编码后的类别，
            e.g. `[[xmin, ymin, xmax, ymax, class], ...]`
        """
        root = ET.parse(file_path).getroot()

        # 图像的尺寸
        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)

        # 提取所有的标签
        target = []
        for obj in root.iter('object'):
            # 样本是否难以预测
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # 归一化方框位置
            points = ['xmin', 'ymin', 'xmax', 'ymax']
            data = []
            for i, pt in enumerate(points):
                pt = int(bbox.find(pt).text) - 1
                pt = pt/w if i % 2 == 0 else pt/h
                data.append(pt)

            data.append(self.class_to_index[name])
            target.append(data)

        return target


class VOCDataset(Dataset):
    """ VOC 数据集 """

    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, root: Union[str, List[str]], image_set: Union[str, List[str]],
                 tranformer: Transformer = None, keep_difficult=False):
        """
        Parameters
        ----------
        root: str or List[str]
            数据集的根路径，下面必须有 `Annotations`、`ImageSets` 和 `JPEGImages` 文件夹

        image_set: str or List[str]
            数据集的种类，可以是 `train`、`val`、`trainval` 或者 `test`

        transformer: Transformer
            数据增强器

        keep_difficulty: bool
            是否保留 difficult 为 1 的样本
        """
        super().__init__()
        if isinstance(root, str):
            root = [root]
        if isinstance(image_set, str):
            image_set = [image_set]
        if len(root) != len(image_set):
            raise ValueError("`root` 和 `image_set` 的个数必须相同")

        self.root = root
        self.image_set = image_set
        self.n_classes = len(self.classes)
        self.keep_difficult = keep_difficult
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        self.transformer = tranformer    # 数据增强器
        self.annotation_transformer = AnnotationTransformer(
            self.class_to_index, keep_difficult)

        # 获取指定数据集中的所有图片和标签文件路径
        self.image_names = []
        self.image_paths = []
        self.annotation_paths = []

        for root, image_set in zip(self.root, self.image_set):
            with open(path.join(root, f'ImageSets/Main/{image_set}.txt')) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.image_names.append(line)
                    self.image_paths.append(
                        path.join(root, f'JPEGImages/{line}.jpg'))
                    self.annotation_paths.append(
                        path.join(root, f'Annotations/{line}.xml'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """ 获取样本

        Parameters
        ----------
        index: int
            下标

        Returns
        -------
        image: Tensor of shape `(3, H, W)`
            增强后的图像数据

        target: `np.ndarray` of shape `(n_objects, 5)`
            标签数据
        """
        image_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]

        # 读入图片和标签数据
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        target = np.array(self.annotation_transformer(annotation_path))
        bbox, label = target[:, :4], target[:, -1]

        # 数据增强
        if self.transformer:
            image, bbox, label = self.transformer.transform(image, bbox, label)
            target = np.hstack((bbox, label[:, np.newaxis]))

        return torch.from_numpy(image).permute(2, 0, 1), target


def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]):
    """ 整理 dataloader 取出的数据

    Parameters
    ----------
    batch: list of shape `(N, 2)`
        一批数据，列表中的每一个元组包括两个元素：
        * image: Tensor of shape `(3, H, W)`
        * target: `~np.ndarray` of shape `(n_objects, 5)`

    Returns
    -------
    image: Tensor of shape `(N, 3, H, W)`
        图像

    target: List[Tensor]
        标签
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images, 0), targets
