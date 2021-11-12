# coding:utf-8
from itertools import product
from math import sqrt

import torch


class PriorBox:
    """ 用来生成先验框的类 """

    def __init__(self, **config):
        """
        Parameters
        ----------
        **config:
            用户自定义的配置
        """
        # 默认配置
        self.config = {
            "image_size": 300,
            'variance': (0.1, 0.2),
            'steps': [8, 16, 32, 64, 100, 300],
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        }
        self.config.update(config)

        self.aspect_ratios = self.config['aspect_ratios']  # 宽高比的种类
        self.feature_maps = self.config['feature_maps']    # 特征图大小
        self.image_size = self.config['image_size']        # 图像大小
        self.min_sizes = self.config['min_sizes']          # s_k*image_size
        self.max_sizes = self.config['max_sizes']          # s_(k+1)*image_size
        self.variance = self.config['variance']
        self.steps = self.config["steps"]           # 原图尺寸/特征图尺寸，可理解为感受野

    def __call__(self):
        """ 得到所有先验框

        Returns
        -------
        boxes: Tensor of shape (n_anchors, 4)
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

