# coding:utf-8
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from .dataset import VOCDataset
from .ssd import SSD


class EvalPipeline:
    """ 测试模型流水线 """

    def __init__(self, model_path: str, dataset: VOCDataset, image_size=300, conf_thresh=0.6, use_gpu=True):
        """
        Parameters
        ----------
        model_path: str
            模型文件路径

        dataset: VOCDataset
            数据集

        use_gpu: bool
            是否使用 GPU

        conf_thresh: float
            置信度阈值
        """
        self.use_gpu = use_gpu
        self.dataset = dataset
        self.conf_thresh = conf_thresh
        self.model = SSD(self.dataset.n_classes+1, image_size=image_size)
        self.model = self.model.to('cuda:0' if use_gpu else 'cpu')
        self.model.load(model_path)

    def eval(self):
        """ 测试模型，获取 mAP """
        # 获取每一种类存在于哪些图片中
        preds = {}

        for image_path, image_name in zip(self.dataset.image_paths, self.dataset.image_names):
            # 读入图片
            x = Image.open(image_path).convert('RGB')
            w, h = x.size
            x = cv.resize(np.array(x), (300, 300)).astype(np.float32)
            x = ToTensor()(x).unsqueeze(0)

            out = self.model.predict(x)[0]

            for c in self.dataset.classes:
                y = out[c]
