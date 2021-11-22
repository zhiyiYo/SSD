# coding:utf-8
from random import choice as randchoice
from typing import List

import cv2 as cv
import numpy as np
import torch
from numpy import ndarray, random
from torchvision import transforms as T

from .box_utils import jaccard_overlap_numpy


class Transformer:
    """ 图像增强接口 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 对输入的图像进行增强

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            图像，图像模式是 RGB 或者 HUV，没有特殊说明默认 RGB 模式

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        raise NotImplementedError("图像增强方法必须被重写")


class Compose(Transformer):
    """ 图像增强器流水线 """

    def __init__(self, transformers: List[Transformer]):
        """
        Parameters
        ----------
        transformers: List[Transformer]
            图像增强器列表
        """
        self.transformers = transformers

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        for t in self.transformers:
            image, bbox, label = t.transform(image, bbox, label)

        return image, bbox, label


class ImageToFloat32(Transformer):
    """ 将图像数据类型转换为 `np.float32` """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        return image.astype(np.float32), bbox, label


class ToTensor(Transformer):
    """ 将 np.ndarray 图像转换为 Tensor """

    def __init__(self, image_size=300, mean=(123, 117, 104)):
        """
        Parameters
        ----------
        image_size: int
            缩放后的图像尺寸

        mean: tuple
            RGB 图像各通道的均值
        """
        super().__init__()
        self.mean = mean
        self.image_size = image_size

    def transform(self, image: ndarray, bbox: ndarray = None, label: ndarray = None):
        """ 将图像进行缩放、中心化并转换为 Tensor

        Parameters
        ----------
        image: `~np.ndarray`
            RGB 图像

        bbox, label: None
            没有用到

        Returns
        -------
        image: Tensor of shape `(1, 3, image_size, image_size)`
            转换后的图像
        """
        size = self.image_size
        x = cv.resize(image, (size, size)).astype(np.float32)
        x -= self.mean
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x


class Centralization(Transformer):
    """ 图像中心化 """

    def __init__(self, mean=(123, 117, 104)):
        super().__init__()
        self.mean = mean

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        image = image.astype(np.float32)
        return image-self.mean, bbox, label


class BBoxToAbsoluteCoords(Transformer):
    """ 将归一化的边界框还原为原始边界框 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, c = image.shape
        bbox[:, [0, 2]] *= w
        bbox[:, [1, 3]] *= h
        return image, bbox, label


class BBoxToPercentCoords(Transformer):
    """ 将未归一化的边界框归一化 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, c = image.shape
        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h
        return image, bbox, label


class RandomSaturation(Transformer):
    """ 随机调整图像饱和度 """

    def __init__(self, lower=0.5, upper=1.5):
        """
        Parameters
        ----------
        lower: float
            饱和度倍数下限

        upper: float
            饱和度倍数上限
        """
        if lower > upper or lower < 0:
            raise ValueError("饱和度倍数上下限非法")

        self.lower = lower
        self.upper = upper

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 调整图像饱和度

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            HSV 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, bbox, label


class RandomHue(Transformer):
    """ 随机调整色调 """

    def __init__(self, delta=18):
        """
        Parameters
        ----------
        delta: float
            调整尺度，要求在 `[0, 360]` 之间
        """
        super().__init__()
        self.delta = np.abs(delta) % 360

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 调整色调

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            HSV 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, bbox, label


class RandomSwapChannel(Transformer):
    """ 随机交换图像的颜色通道 """

    def __init__(self):
        super().__init__()
        self.channel_orders = [
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0)
        ]

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        if random.randint(2):
            order = randchoice(self.channel_orders)
            image = image[:, :, order]

        return image, bbox, label


class ConvertColor(Transformer):
    """ 改变图像模式 """

    def __init__(self, current='RGB', to='HSV'):
        """
        Parameters
        ----------
        current: str
            当前颜色模式

        to: str
            目标颜色模式
        """
        super().__init__()
        if current == to:
            raise ValueError("图像转换前后的模式不允许相同")
        self.mode = to

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 改变图像模式

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 或者 HSV 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        mode = cv.COLOR_RGB2HSV if self.mode == 'RGB' else cv.COLOR_HSV2RGB
        image = cv.cvtColor(image, mode)
        return image, bbox, label


class RandomContrast(Transformer):
    """ 随机调整对比度 """

    def __init__(self, lower=0.8, upper=1.1):
        """
        Parameters
        ----------
        lower: float
            饱和度倍数下限

        upper: float
            饱和度倍数上限
        """
        if lower > upper or lower < 0:
            raise ValueError("饱和度倍数上下限非法")

        self.lower = lower
        self.upper = upper

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        if random.randint(2):
            image *= random.uniform(self.lower, self.upper)

        return image, bbox, label


class RandomBrightness(Transformer):
    """ 随机调整图像亮度 """

    def __init__(self, delta=32):
        """
        Parameters
        ----------
        delta: float
            亮度变化范围
        """
        self.delta = np.abs(delta) % 255

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        if random.randint(2):
            image += random.uniform(-self.delta, self.delta)

        return image, bbox, label


class RandomSampleCrop(Transformer):
    """ 随机裁剪 """

    def __init__(self):
        super().__init__()
        self.sample_options = [
            # 直接返回原图
            None,
            # 随机裁剪，裁剪区域和边界框的交并比有阈值要求
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # 随机裁剪
            (None, None),
        ]

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, _ = image.shape

        while True:
            mode = randchoice(self.sample_options)

            # 直接返回原图
            if mode is None:
                return image, bbox, label

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 最多尝试 50 次，避免死循环
            for _ in range(50):

                # 随机选取采样区域的宽高
                ww = random.uniform(0.3*w, w)
                hh = random.uniform(0.3*h, h)

                # 要求宽高比在 0.5 ~ 2 之间
                if not 0.5 <= hh/ww <= 2:
                    continue

                # patch 的四个坐标
                left = random.uniform(high=w-ww)
                top = random.uniform(high=h-hh)
                rect = np.array([left, top, left+ww, top+hh], dtype=np.int)

                # 交并比不满足阈值条件就舍弃这个 patch
                iou = jaccard_overlap_numpy(rect, bbox)
                if iou.min() > max_iou or iou.max() < min_iou:
                    continue

                # 裁剪下 patch
                patch = image[rect[1]:rect[3], rect[0]:rect[2]]

                # 判断边界框的中心有没有落在 patch 里面
                centers = (bbox[:, :2]+bbox[:, 2:])/2
                m1 = (centers[:, 0] > rect[0]) & (centers[:, 1] > rect[1])
                m2 = (centers[:, 0] < rect[2]) & (centers[:, 1] < rect[3])
                mask = m1 & m2

                # 如果没有任何一个边界框的中心在 patch 里面就舍弃这个 patch
                if not mask.any():
                    continue

                # 中心落在 patch 里面的边界框及其标签
                bbox_ = bbox[mask].copy()
                label_ = label[mask]

                # 对 patch 里面的边界框进行坐标平移，使其以 patch 的左上角为原点
                bbox_[:, :2] = np.clip(bbox_[:, :2]-rect[:2], 0, np.inf)
                bbox_[:, 2:] = np.clip(
                    bbox_[:, 2:]-rect[:2], 0, rect[2:]-rect[:2]-1)

                return patch, bbox_, label_


class Expand(Transformer):
    """ 增大图像并在旁边填充颜色 """

    def __init__(self, mean: tuple):
        """
        Parameters
        ----------
        mean: tuple
            填充的颜色
        """
        self.mean = mean

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        if random.randint(2):
            return image, bbox, label

        h, w, c = image.shape
        ratio = random.uniform(1, 4)
        left = int(random.uniform(0, w*ratio-w))
        top = int(random.uniform(0, h*ratio-h))

        # 扩充图像
        expand_image = np.zeros(
            (int(h*ratio), int(w*ratio), c), image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[top:top+h, left:left+w] = image

        # 平移边界框的原点为拓展图像的左上角
        bbox[:, :2] += [left, top]
        bbox[:, 2:] += [left, top]

        return expand_image, bbox, label


class Resize(Transformer):
    """ 调整图像大小 """

    def __init__(self, size=(300, 300)):
        super().__init__()
        self.size = size

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 调整图像大小

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            已经归一化的边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        # 归一化后的边界框无需做出任何改变
        return cv.resize(image, self.size), bbox, label


class RandomFlip(Transformer):
    """ 随机翻转 """

    def __init__(self):
        super().__init__()
        self.direction = [None, 'H', 'V']   # 翻转方向

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, c = image.shape
        direction = randchoice(self.direction)

        if direction == 'H':
            image = np.fliplr(image)
            bbox[:, 0::2] = w-bbox[:, 2::-2]
        elif direction == 'V':
            image = np.flipud(image)
            bbox[:, 1::2] = h-bbox[:, 3::-2]

        return image, bbox, label


class ColorJitter(Transformer):
    """ 随机调整图像的色调、饱和度、对比度、亮度和颜色通道 """

    def __init__(self):
        super().__init__()
        self.transformers = [
            RandomContrast(),
            ConvertColor(current='RGB', to='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', to='RGB'),
            RandomContrast()
        ]

        self.rand_brightness = RandomBrightness()
        self.rand_swap_channel = RandomSwapChannel()

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        image, bbox, label = self.rand_brightness.transform(image, bbox, label)

        if random.randint(2):
            t = Compose(self.transformers[:-1])
        else:
            t = Compose(self.transformers[1:])

        return self.rand_swap_channel.transform(*t.transform(image, bbox, label))


class SSDAugmentation(Transformer):
    """ SSD 神经网络训练时使用的数据增强器 """

    def __init__(self, image_size=300, mean=(123, 117, 104)):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            ColorJitter(),
            Expand(mean),
            RandomSampleCrop(),
            RandomFlip(),
            BBoxToPercentCoords(),
            Resize((image_size, image_size)),
            Centralization(mean)
        ])

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 图像增强


        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        return self.transformers.transform(image, bbox, label)
