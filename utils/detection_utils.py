# coding:utf-8
from typing import List

import cv2 as cv
import numpy as np
from imutils.video import FPS, WebcamVideoStream
from net import SSD


def image_detect(model_path: str, image_path: str, classes: List[str], conf_thresh=0.6, use_gpu=True):
    """ 检测图像中的目标

    Parameters
    ----------
    model_path: str
        模型路径

    image_path: str
        图片路径

    classes: List[str]
        类别列表

    conf_thresh: float
        置信度阈值，不会显示小于这个阈值的预测框

    use_gpu: bool
        是否使用 gpu 加速检测

    Returns
    -------
    image: `~PIL.Image.Image`
        绘制了边界框、类别和置信度的图像
    """
    # 创建模型
    model = SSD(len(classes)+1)
    if use_gpu:
        model = model.cuda()

    # 载入模型
    model.load(model_path)
    model.eval()

    # 检测目标
    return model.detect(image_path, classes, conf_thresh, use_gpu=use_gpu)


def camera_detect(model_path: str, classes: List[str], camera_src=0, conf_thresh=0.6, use_gpu=True):
    """ 从摄像头中实时检测物体

    Parameters
    ----------
    model_path: str
        模型路径

    classes: List[str]
        类别列表

    camera_src: int
        摄像头源，0 代表默认摄像头

    conf_thresh: float
        置信度阈值，不会显示小于这个阈值的预测框

    use_gpu: bool
        是否使用 gpu 加速检测
    """
    # 创建模型
    model = SSD(len(classes)+1)
    if use_gpu:
        model = model.cuda()

    # 载入模型
    model.load(model_path)
    model.eval()

    # 创建帧率统计器
    fps = FPS().start()

    print('📸 正在检测物体中，按 q 退出...')

    # 打开摄像头
    stream = WebcamVideoStream(src=camera_src).start()
    while True:
        image = stream.read()
        image = np.array(model.detect(
            image, classes, conf_thresh, use_gpu=use_gpu))
        fps.update()

        # 显示检测结果
        cv.imshow('camera detection', image)

        # 退出程序
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print(f'检测结束，帧率：{fps.fps()} FPS')