# coding:utf-8
from typing import Iterable
from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpMultiStepLR(_LRScheduler):
    """ 热启动学习率规划器 """

    def __init__(self, optimizer: Optimizer, milestones: Iterable[int], gamma=0.1, warm_up_factor=1/3, warm_up_iters=500, last_epoch=-1):
        """
        Parameters
        ----------
        optimizer: Optimizer
            优化器

        milestones: Iterable[int]
            学习率退火的节点

        gamma: float
            学习率衰减率

        warm_up_factor: float
            第一次迭代时热启动学习率和初始学习率的比值

        warm_up_iters: int
            热启动迭代次数

        last_epoch: int
            初始迭代次数，-1 代表从零开始训练
        """
        self.gamma = gamma
        self.warm_up_factor = warm_up_factor
        self.warm_up_iters = warm_up_iters
        self.milestones = sorted(milestones)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """ 根据迭代次数获取学习率 """
        e = self.last_epoch

        if e < self.warm_up_iters:
            # 热启动期间学习率线性增长
            delta = (1-self.warm_up_factor)*e/self.warm_up_iters
            factor = self.warm_up_factor + delta
        else:
            factor = 1

        # 学习率衰减率
        gamma = self.gamma**bisect_right(self.milestones, e)

        lrs = [lr*factor*gamma for lr in self.base_lrs]
        return lrs