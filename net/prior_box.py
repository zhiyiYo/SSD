# coding:utf-8
from itertools import product
from math import sqrt

import torch


class PriorBox:
    """ 用来生成先验框的类 """

    def __init__(self, image_size=300, feature_maps: list = None, min_sizes: list = None,
                 max_sizes: list = None, aspect_ratios: list = None, steps: list = None, **kwargs):
        """
        Parameters
        ----------
        image_size: int
            图像大小

        feature_maps: list
            特征图大小

        min_sizes: list
            特征图中的最小正方形先验框的尺寸

        max_sizes: list
            下一个特征图中的最小正方形先验框的尺寸

        aspect_ratios: list
            长宽比

        steps: list
            步长，可理解为感受野大小
        """
        self.image_size = image_size
        self.feature_maps = feature_maps or [38, 19, 10, 5, 3, 1]
        self.min_sizes = min_sizes or [30, 60, 111, 162, 213, 264]
        self.max_sizes = max_sizes or [60, 111, 162, 213, 264, 315]
        self.steps = steps or [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = aspect_ratios or [
            [2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def __call__(self):
        """ 得到所有先验框

        Returns
        -------
        boxes: Tensor of shape `(n_priors, 4)`
            先验框
        """
        boxes = []

        for k, f in enumerate(self.feature_maps):
            f_k = self.image_size / self.steps[k]

            for i, j in product(range(f), repeat=2):
                # 中心坐标，向右为 x 轴正方向，向下为 y 轴正方向
                cx = (j+0.5) / f_k
                cy = (i+0.5) / f_k

                # 1 和 1'
                s_k = self.min_sizes[k]/self.image_size
                s_k_prime = sqrt(s_k*self.max_sizes[k]/self.image_size)

                boxes.append([cx, cy, s_k, s_k])
                boxes.append([cx, cy, s_k_prime, s_k_prime])

                # 根据其余的 ar 计算宽和高
                for ar in self.aspect_ratios[k]:
                    boxes.append([cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)])
                    boxes.append([cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)])

        boxes = torch.Tensor(boxes).clamp(min=0, max=1)
        return boxes
