# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'test')

model_path = 'model/SSD_120000.pth'
eval_pipeline = EvalPipeline(model_path, dataset, conf_thresh=0.001, cache=False)
eval_pipeline.eval()