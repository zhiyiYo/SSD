# coding:utf-8
from net import VOCDataset, TrainPipeline
from utils.augmentation_utils import SSDAugmentation

# load dataset
root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'trainval', SSDAugmentation(), True)

# train
train_pipeline = TrainPipeline(
    dataset,
    vgg_path='model/vgg16_reducedfc.pth',
    ssd_path='model/SSD_20480.pth',
    batch_size=16,
    start_iter=20480,
    log_file='log/losses_20480.json',
)

train_pipeline.train()
