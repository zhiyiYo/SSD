# coding:utf-8
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """ 损失函数 """

    def __init__(self, n_classes: int, variance=(0.1, 0.2), overlap_thresh=0.5, neg_pos_ratio=3, use_gpu=True):
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

    def forward(self, pred: Tuple[Tensor, Tensor, Tensor], target: Tensor):
        """ 计算损失

        Parameters
        ----------
        pred: Tuple[Tensor]
            SSD 网络的预测结果

        target: Tensor of shape (N, n_objects, 5)
            标签，包含边界框位置和类别，每张图中可能不止有一个目标
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
        positive_index = positive.unsqueeze(-1).expand_as(loc_pred)

        # 方框位置损失
        loc_positive = loc_pred[positive_index].view(-1, 4)
        loc_t = loc_t[positive_index].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_positive, loc_t)

        # 困难样本挖掘，conf_logP 的 shape: (N*n_priors, 1)
        batch_conf_pred = conf_pred.view(-1, self.n_classes)
        conf_logP = log_sum_exp(
            batch_conf_pred)-batch_conf_pred.gather(1, conf_t.type(torch.int64).view(-1, 1))

        # 去除正样本，根据损失获取负样本的损失排名，conf_logP 的 shape: (N, n_priors)
        conf_logP = conf_logP.view(N, -1)
        conf_logP[positive] = 0
        _, loss_index = conf_logP.sort(dim=1, descending=True)
        _, loss_rank = loss_index.sort(dim=1)

        # 根据 负样本:正样本 选取出 top_k 个负样本
        n_positive = positive.long().sum(dim=1, keepdim=True)
        n_negative = torch.clamp(
            self.neg_pos_ratio*n_positive, max=positive.size(1)-1)
        # 负样本标记，shape: (N, n_priors)
        negative = loss_rank < n_negative.expand_as(loss_rank)

        # 置信度损失
        index = (positive+negative).unsqueeze(2).expand_as(conf_pred).gt(0)
        conf_pred = conf_pred[index].view(-1, self.n_classes)
        conf_t = conf_t[(positive+negative) > 0]
        conf_loss = F.cross_entropy(conf_pred, conf_t.type(torch.int64))

        # 将损失除以正样本个数
        n_positive = n_negative.detach().sum()
        loc_loss /= n_positive
        conf_loss /= n_positive

        return loc_loss, conf_loss
