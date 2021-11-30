# coding:utf-8
import json

import numpy as np
import matplotlib.pyplot as plt

from .log_utils import LossLogger


def plot_loss(log_file: str, epoch_size: int = None):
    """ 绘制损失曲线

    Parameters
    ----------
    log_file: str
        损失日志文件路径3

    epoch_size: int
        每一个 epoch 需要迭代多少次，如果为 `None`，横坐标将是 epoch 而不是 iterations
    """
    logger = LossLogger(None, log_file)
    epoch_size = 1 if not epoch_size else epoch_size
    iteration = np.arange(1, len(logger.losses)+1)*epoch_size
    xlabel = 'epoch' if epoch_size == 1 else 'iterations'

    fig, ax = plt.subplots(1, 1, num='损失曲线')
    ax.plot(iteration, logger.losses, label='Total Loss')
    ax.plot(iteration, logger.conf_losses, label='Confidence Loss')
    ax.plot(iteration, logger.loc_losses, label='Location Loss')
    ax.set(xlabel=xlabel, ylabel='loss', title='Loss Curve')
    ax.legend()

    return fig, ax


def plot_PR(file_path: str, class_name: str):
    """ 绘制 PR 曲线

    Parameters
    ----------
    file_path: str
        由 eval.py 生成的测试结果文件路径

    class_name: str
        类别名
    """
    with open(file_path, encoding='utf-8') as f:
        result = json.load(f)[class_name]

    fig, ax = plt.subplots(1, 1, num='PR 曲线')
    ax.plot(result['recall'], result['precision'])
    ax.set(xlabel='recall', ylabel='precision', title='PR curve')
    return fig, ax


def plot_AP(file_path: str):
    """ 绘制 AP 柱状图 """
    with open(file_path, encoding='utf-8') as f:
        result = json.load(f)

    AP = []
    classes = []
    for k, v in result.items():
        if k!='mAP':
            AP.append(v['AP'])
            classes.append(k)

    fig, ax = plt.subplots(1, 1, num='AP 柱状图')
    ax.barh(range(len(AP)), AP, height=0.6, tick_label=classes)
    ax.set(xlabel='AP', title=f'mAP: {result["mAP"]:.2%}')

    return fig, ax
