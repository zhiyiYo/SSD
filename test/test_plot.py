# coding:utf-8
import unittest
from utils.plot_utils import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc_file('resource/theme/matlab.mplstyle')


class TestPlotUtils(unittest.TestCase):
    """ 测试绘图工具 """

    def test_plot_loss(self):
        """ 测试损失绘制 """
        fig, ax = plot_loss('log/losses_42514.json')
        plt.show()

    def test_plot_PR(self):
        """ 测试 PR 曲线绘制 """
        fig, ax = plot_PR('eval/SSD.json', 'cat')
        plt.show()

    def test_plot_AP(self):
        """ 测试 AP 柱状图绘制 """
        fig, ax = plot_AP('eval/SSD.json')
        plt.show()
