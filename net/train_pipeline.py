# coding:utf-8
import itertools
import math
import os
import traceback
from datetime import datetime

import torch
from torch import nn, optim
from torch.nn import init

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.log_utils import LossLogger

from .dataset import collate_fn
from .loss import SSDLoss
from .ssd import SSD


def exception_handler(train_func):
    """ 处理训练过程中发生的异常并保存模型 """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            train_pipeline.save()
            exit()

    return wrapper


class TrainPipeline:
    """ 训练 SSD 模型 """

    def __init__(self, dataset: Dataset, vgg_path: str = None, ssd_path: str = None,
                 lr=0.001, momentum=0.9, weight_decay=5e-4, lr_steps=(40000, 50000, 6000),
                 batch_size=16, start_iter=0, max_iter=60000, use_gpu=True, save_dir='model',
                 log_file: str = None, log_dir='log', **config):
        """
        Parameters
        ----------
        dataset: Dataset
            训练数据集

        vgg_path: str
            预训练的 VGG16 模型文件路径

        ssd_path: Union[str, None]
            SSD 模型文件路径，有以下两种选择:
            * 如果不为 `None`，将使用模型文件中的参数初始化 `SSD`
            * 如果为 `None`，将使用 `init.xavier` 方法初始化 VGG16 之后的卷积层参数

        lr: float
            学习率

        momentum: float
            冲量

        weight_decay: float
            权重衰减

        lr_steps: Tuple[int]
            学习率退火的节点

        batch_size: int
            训练集 batch 大小

        start_iter: int
            SSD 模型文件包含的参数是训练了多少次的结果

        max_iter: int
            最多迭代多少次

        use_gpu: bool
            是否使用 GPU 加速训练

        save_dir: str
            保存 SSD 模型的文件夹

        log_file: str
            训练损失数据历史记录文件，要求是 json 文件

        save_dir: str
            训练损失数据保存的文件夹

        **config:
            先验框生成、先验框和边界框匹配以及 NMS 算法的配置
        """
        self.config = {
            # 通用配置
            'n_classes': 21,
            'variance': (0.1, 0.2),

            # 先验框生成配置
            "image_size": 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

            # NMS 配置
            'top_k': 200,
            'nms_thresh': 0.45,
            'conf_thresh': 0.01,

            # 先验框和边界框匹配配置
            'overlap_thresh': 0.5,

            # 困难样本挖掘配置
            'neg_pos_ratio': 3,
        }
        self.config.update(config)

        self.dataset = dataset
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.batch_size = batch_size

        # 一个 epoch 有多少个 batch
        self.n_batches = math.floor(len(self.dataset)/self.batch_size)+1

        # 创建模型
        self.model = SSD(**self.config).to(self.device)

        # 迭代次数
        self.current_iter = start_iter
        self.start_iter = start_iter
        self.max_iter = max_iter

        # 损失函数和优化器
        self.critorion = SSDLoss(**self.config)
        self.optimizer = optim.SGD(self.model.parameters(), lr,
                                   momentum, weight_decay=weight_decay)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, lr_steps, 0.1)

        # 训练损失记录器
        self.logger = LossLogger(self.n_batches, log_file, log_dir)

        # 初始化模型
        if ssd_path:
            self.model.load_state_dict(torch.load(ssd_path))
            print('🧪 成功载入 SSD 模型：' + ssd_path)
        elif vgg_path:
            self.model.vgg.load_state_dict(torch.load(vgg_path))
            self.model.extras.apply(self.xavier)
            self.model.confs.apply(self.xavier)
            self.model.locs.apply(self.xavier)
            print('🧪 成功载入 VGG16 模型：' + vgg_path)
        else:
            raise ValueError("必须指定预训练的 VGG16 模型文件路径")

    def save(self):
        """ 保存模型和训练损失数据 """
        os.makedirs(self.save_dir, exist_ok=True)

        # 保存模型
        self.model.eval()
        path = f'{self.save_dir}/SSD_{self.current_iter}.pth'
        torch.save(self.model.state_dict(), path)

        # 保存训练损失数据
        self.logger.save(f'losses_{self.current_iter}')

        print(f'🎉 已将当前模型保存到 {os.path.join(os.getcwd(), path)}')

    @staticmethod
    def xavier(module):
        """ 使用 xavier 方法初始化模型的参数 """
        if not isinstance(module, nn.Conv2d):
            return

        init.xavier_uniform_(module.weight)
        init.constant_(module.bias, 0)

    @exception_handler
    def train(self):
        """ 训练模型 """
        torch.autograd.set_detect_anomaly(True)

        data_loader = DataLoader(
            self.dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        # 无穷迭代器
        data_iter = itertools.cycle(data_loader)

        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 开始训练！')

        with tqdm(total=self.max_iter-self.start_iter, bar_format=bar_format) as bar:
            self.model.train()

            for i in range(self.start_iter, self.max_iter):
                self.current_iter = i
                if (i-self.start_iter) % self.n_batches == 0:
                    start_time = datetime.now()

                bar.set_description(f"\33[36m🌌 Iteration {i:5d}")

                # 取出数据
                images, targets = next(data_iter)

                # 预测边界框、置信度和先验框
                pred = self.model(images.to(self.device))

                # 计算损失并将误差反向传播
                self.optimizer.zero_grad()
                loc_loss, conf_loss = self.critorion(pred, targets)
                loss = loc_loss + conf_loss  # type:torch.Tensor
                loss.backward()
                self.optimizer.step()
                self.lr_schedule.step()

                cost_time = datetime.now() - start_time
                bar.set_postfix_str(
                    f'训练损失：{loss.item():.5f}, 执行时间：{cost_time}\33[0m')
                bar.update()

                # 更新损失
                self.logger.update(loc_loss.item(), conf_loss.item())

        self.save()
