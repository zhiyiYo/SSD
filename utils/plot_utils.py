# coding:utf-8
import json

import numpy as np
import matplotlib.pyplot as plt

from .log_utils import LossLogger


def plot_loss(log_file: str):
    """ 绘制损失曲线

    Parameters
    ----------
    log_file: str
        损失日志文件路径
    """
    logger = LossLogger(None, log_file)
    iteration = np.arange(0, len(logger.losses))*418

    fig, ax = plt.subplots(1, 1)
    ax.plot(iteration, logger.losses, label='loss')
    ax.plot(iteration, logger.loc_losses, label='loc_loss')
    ax.plot(iteration, logger.conf_losses, label='conf_loss')
    ax.set(xlabel='iteration', ylabel='loss', title='SSD Loss Curve')
    ax.legend()

    return fig, ax
