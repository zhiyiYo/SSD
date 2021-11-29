# coding:utf-8
from net import VOCDataset
from utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/SSD_120000.pth'
image_path = 'resource/image/硝子.png'

# 检测目标
image = image_detect(model_path, image_path, VOCDataset.classes, conf_thresh=0.4)
image.show()