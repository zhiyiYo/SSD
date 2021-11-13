# coding:utf-8
import unittest
from utils.plot_utils import plot_loss

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc_file('resource/matlab.mplstyle')


class TestPlotUtils(unittest.TestCase):
    """ 测试绘图工具 """

    def test_plot_loss(self):
        """ 测试损失绘制 """
        fig, ax = plot_loss('log/losses_20480.json')
        plt.show()
