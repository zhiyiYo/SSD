# coding:utf-8
from net import SSD, VOCDataset


# 创建模型
model = SSD(21).cuda()
model.load('model/SSD_313.pth')
model.eval()

# 检测目标
image_path = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
image = model.detect(image_path, VOCDataset.classes, 0.6)
image.show()