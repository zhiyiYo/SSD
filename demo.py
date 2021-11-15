# coding:utf-8
from net import SSD, VOCDataset


# 模型文件
model_path = 'model/SSD.pth'
image_path = 'resource/image/硝子.png'

# 创建模型
model = SSD(n_classes=21).cuda()
model.load(model_path)
model.eval()

# 检测目标
image = model.detect(image_path, VOCDataset.classes, 0.6)
image.show()