# coding:utf-8
from net import SSD, VOCDataset


# 模型文件
model_path = 'model/SSD_368.pth'
image_path = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'

# 创建模型
model = SSD(n_classes=21, conf_thresh=0.01).cuda()
model.load(model_path)
model.eval()

# 检测目标
image = model.detect(image_path, VOCDataset.classes, 0.6)
image.show()