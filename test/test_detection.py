# coding:utf-8
import unittest

from net import VOCDataset
from utils.detection_utils import *


class TestDetection(unittest.TestCase):
    """ 测试目标检测 """

    def test_image_detect(self):
        """ 测试图像中的目标检测 """
        model_path = 'model/SSD.pth'
        image_path = 'resource/image/硝子.png'

        # 检测目标
        image = image_detect(model_path, image_path, VOCDataset.classes)
        image.show()

    def test_camera_detect(self):
        """ 测试摄像头中的目标检测 """
        model_path = 'model/SSD.pth'
        camera_detect(model_path, VOCDataset.classes, conf_thresh=0.5)
