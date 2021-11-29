# coding:utf-8
import torch
from torch import Tensor

from utils.box_utils import decode, nms


class Detector:
    """ 用于处理 SSD 网络输出的探测器类，在测试时起作用 """

    def __init__(self, n_classes: int, variance: list, top_k=200, conf_thresh=0.01, nms_thresh=0.45) -> None:
        """
        Parameters
        ----------
        n_classes: int
            类别数，包括背景

        variance: Tuple[float, float]
            先验框方差

        top_k: int
            预测框数量的上限

        conf_thresh: float
            置信度阈值

        nms_thresh: float
            nms 操作中 iou 的阈值，越小保留的预测框越少
        """
        self.n_classes = n_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.top_k = top_k

    def __call__(self, loc: Tensor, conf: Tensor, prior: Tensor):
        """ 生成预测框

        Parameters
        ----------
        loc: Tensor of shape `(N, n_priors, 4)`
            预测的偏移量

        conf: Tensor of shape `(N, n_priors, n_classes)`
            类别置信度，需要被 softmax 处理过

        prior: Tensor of shape `(n_priors, 4)`
            先验框

        Returns
        -------
        out: Tensor of shape `(N, n_classes, top_k, 5)`
            检测结果，最后一个维度的前四个元素为边界框的坐标 `(xmin, ymin, xmax, ymax)`，最后一个元素为置信度
        """
        N = loc.size(0)

        # 一张图中可能有多个相同类型的物体，所以多一个 n_classes 维度
        out = torch.zeros(N, self.n_classes, self.top_k, 5)

        for i in range(N):
            # 解码出边界框
            bbox = decode(loc[i], prior, self.variance)
            conf_scores = conf[i].clone()   # Shape: (n_priors, n_classes)

            for c in range(1, self.n_classes):
                # 将置信度小于阈值的置信度元素滤掉
                mask = conf_scores[:, c] > self.conf_thresh
                scores = conf_scores[:, c][mask]    # Shape: (n_prior', )

                # 如果所有的先验框都没有预测出这个类，就直接跳过
                if scores.size(0) == 0:
                    continue

                # 将置信度小于阈值的边界框滤掉
                boxes = bbox[mask]

                # 非极大值抑制，将多余的框滤除
                indexes = nms(boxes, scores, self.nms_thresh, self.top_k)
                out[i, c, :len(indexes)] = torch.cat(
                    (boxes[indexes], scores[indexes].unsqueeze(1)), dim=1)

        return out
