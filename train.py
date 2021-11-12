# coding:utf-8
from net import VOCDataset, TrainPipeline
from utils.augmentation_utils import SSDAugmentation

# load dataset
root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'trainval', SSDAugmentation())

# train
train_pipeline = TrainPipeline(dataset, 'model/vgg16_reducedfc.pth')
train_pipeline.train()