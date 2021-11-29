# coding:utf-8
from typing import Tuple, List

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from utils.box_utils import match, hard_negative_mining


class SSDLoss(nn.Module):
    """ 损失函数 """

    def __init__(self, n_classes: int, variance=(0.1, 0.2), overlap_thresh=0.5, neg_pos_ratio=3, use_gpu=True, **kwargs):
        """
        Parameters
        ----------
        n_classes: int
            类别数，包括背景

        variance: Tuple[float, float]
            先验框方差

        overlap_thresh: float
            IOU 阈值，默认为 0.5

        neg_pos_ratio: int
            负样本和正样本的比例，默认 3:1

        use_gpu: bool
            是否使用 gpu
        """
        super().__init__()

        if len(variance) != 2:
            raise ValueError("variance 只能有 2 元素")

        self.use_gpu = use_gpu
        self.variance = variance
        self.n_classes = n_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.overlap_thresh = overlap_thresh

    def forward(self, pred: Tuple[Tensor, Tensor, Tensor], target: List[Tensor]):
        """ 计算损失

        Parameters
        ----------
        pred: Tuple[Tensor]
            SSD 网络的预测结果，包含以下数据：
            * loc: Tensor of shape `(N, n_priors, 4)`
            * conf: Tensor of shape `(N, n_priors, n_classes)`
            * prior: Tensor of shape `(n_priors, 4)`

        target: list of shape `(N, )`
            标签列表，每个标签的形状为 `(n_objects, 5)`，包含边界框位置和类别，每张图中可能不止有一个目标
        """
        loc_pred, conf_pred, prior = pred
        N = conf_pred.size(0)
        n_priors = prior.size(0)

        # 将先验框和边界框 ground truth 匹配，loc_t 保存编码后的偏移量
        loc_t = torch.Tensor(N, n_priors, 4)
        conf_t = torch.Tensor(N, n_priors)
        prior = prior.detach()
        for i in range(N):
            bbox = target[i][:, :-1].detach()
            label = target[i][:, -1].detach()
            loc_t[i], conf_t[i] = match(
                self.overlap_thresh, prior, bbox, self.variance, label)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # 正样本标记，索引的 shape: (N, n_priors, 4)，会将所有正样本选出来合成为一维向量
        positive = conf_t > 0   # Shape: (N, n_priors)
        pos_mask = positive.unsqueeze(positive.dim()).expand_as(loc_pred)

        # 方框位置损失
        loc_positive = loc_pred[pos_mask].view(-1, 4)
        loc_t = loc_t[pos_mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_positive, loc_t, reduction='sum')

        # 困难样本挖掘
        mask = hard_negative_mining(conf_pred, conf_t, self.neg_pos_ratio)

        # 置信度损失
        conf_pred = conf_pred[mask].view(-1, self.n_classes)
        conf_t = conf_t[mask].type(torch.int64)
        conf_loss = F.cross_entropy(conf_pred, conf_t, reduction='sum')

        # 将损失除以正样本个数
        n_positive = loc_positive.size(0)
        loc_loss /= n_positive
        conf_loss /= n_positive

        return loc_loss, conf_loss
