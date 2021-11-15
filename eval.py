# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'test')

model_path = 'model/history/SSD_42480.pth'
eval_pipeline = EvalPipeline(model_path, dataset, use_07_metric=False)
eval_pipeline.eval()