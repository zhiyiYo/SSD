# coding: utf-8
from typing import Union
from itertools import cycle

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from PIL import Image, ImageDraw, ImageFont, ImageColor


def log_sum_exp(x: Tensor):
    """ 计算 log(∑exp(xi))

    Parameters
    ----------
    x: Tensor of shape `(n, n_classes)`
    """
    x_max = x.detach().max()
    out = torch.log(torch.sum(torch.exp(x-x_max), dim=1, keepdim=True))+x_max
    return out


def jaccard_overlap(prior: Tensor, bbox: Tensor):
    """ 计算预测的先验框和边界框真值的交并比，四个坐标为 `(xmin, ymin, xmax, ymax)`

    Parameters
    ----------
    prior: Tensor of shape `(A, 4)`
        先验框

    bbox: Tensor of shape  `(B, 4)`
        边界框真值

    Returns
    -------
    iou: Tensor of shape `(A, B)`
        交并比
    """
    A = prior.size(0)
    B = bbox.size(0)

    # 将先验框和边界框真值的 xmax、ymax 以及 xmin、ymin进行广播使得维度一致，(A, B, 2)
    # 再计算 xmax 和 ymin 较小者、xmin 和 ymin 较大者，W=xmax较小-xmin较大，H=ymax较小-ymin较大
    xy_max = torch.min(prior[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox[:, 2:].broadcast_to(A, B, 2))
    xy_min = torch.max(prior[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox[:, :2].broadcast_to(A, B, 2))

    # 计算交集面积
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[:, :, 0]*inter[:, :, 1]

    # 计算每个矩形的面积
    area_prior = ((prior[:, 2]-prior[:, 0]) *
                  (prior[:, 3]-prior[:, 1])).unsqueeze(1).expand(A, B)
    area_bbox = ((bbox[:, 2]-bbox[:, 0]) *
                 (bbox[:, 3]-bbox[:, 1])).broadcast_to(A, B)

    return inter/(area_prior+area_bbox-inter)


def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和其他边界框的交并比

    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框

    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes+inter)  # type: np.ndarray
    return iou


def center_to_corner(boxes: Tensor):
    """ 将 `(cx, cy, w, h)` 形式的边界框变换为 `(xmin, ymin, xmax, ymax)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)


def corner_to_center(boxes: Tensor):
    """ 将 `(xmin, ymin, xmax, ymax)` 形式的边界框变换为 `(cx, cy, w, h)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]), dim=1)


def encode(prior: Tensor, matched_bbox: Tensor, variance: tuple):
    """ 编码先验框和与边界框之间的偏置量

    Parameters
    ----------
    prior: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    matched_bbox: Tensor of shape `(n_priors, 4)`
        匹配到的边界框，坐标形式为 `(xmin, ymin, xmax, ymax)`

    variance: Tuple[float, float]
        先验框方差

    Returns
    -------
    g: Tensor of shape `(n_priors, 4)`
        编码后的偏置量
    """
    matched_bbox = corner_to_center(matched_bbox)
    g_cxcy = (matched_bbox[:, :2]-prior[:, :2]) / (variance[0]*prior[:, 2:])
    g_wh = torch.log(matched_bbox[:, 2:]/prior[:, 2:]+1e-5) / variance[1]
    return torch.cat((g_cxcy, g_wh), dim=1)


def decode(loc: Tensor, prior: Tensor, variance: tuple):
    """ 根据偏移量和先验框位置解码出边界框的位置

    Parameters
    ----------
    loc: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    prior: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    variance: Tuple[float, float]
        先验框方差

    Returns
    -------
    g: Tensor of shape `(n_priors, 4)`
        边界框的位置
    """
    bbox = torch.cat((
        prior[:, :2] + prior[:, 2:] * variance[0] * loc[:, :2],
        prior[:, 2:] * torch.exp(variance[1] * loc[:, 2:])), dim=1)
    bbox = center_to_corner(bbox)
    return bbox


def match(overlap_thresh: float, prior: Tensor, bbox: Tensor, variance: tuple, label: Tensor):
    """ 匹配先验框和边界框真值

    Parameters
    ----------
    overlap_thresh: float
        IOU 阈值

    prior: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    bbox: Tensor of shape `(n_objects, 4)`
        边界框真值，坐标形式为 `(xmin, ymin, xmax, ymax)`

    variance: Tuple[float, float]
        先验框方差

    label: Tensor of shape `(n_objects, )`
        类别标签

    Returns
    -------
    loc: Tensor of shape `(n_priors, 4)`
        编码后的先验框和边界框的位置偏移量

    conf: Tensor of shape `(n_priors, )`
        先验框中的物体所属的类
    """
    iou = jaccard_overlap(center_to_corner(prior), bbox)

    # 获取和每个边界框匹配地最好的先验框的 iou 和索引，返回值形状 (n_objects, )
    best_prior_iou, best_prior_index = iou.max(dim=0)

    # 获取和每个先验框匹配地最好的边界框的 iou 和索引，返回值形状为 (n_priors, )
    best_bbox_iou, best_bbox_index = iou.max(dim=1)

    # 边界框匹配到的先验框不能再和别的边界框匹配，即使 iou 小于阈值也必须匹配，所以填充一个大于1的值
    best_bbox_iou.index_fill_(0, best_prior_index, 2)
    for i in range(len(best_prior_index)):
        best_bbox_index[best_prior_index[i]] = i

    # 挑选出和先验框匹配的边界框，形状为 (n_priors, 4)
    matched_bbox = bbox[best_bbox_index]

    # 标记先验框中的物体所属的类，形状为 (n_priors, )，+1 是为了让出背景类的位置
    conf = label[best_bbox_index]+1
    conf[best_bbox_iou < overlap_thresh] = 0

    # 对先验框的位置进行编码
    loc = encode(prior, matched_bbox, variance)

    return loc, conf


def nms(boxes: Tensor, scores: Tensor, overlap_thresh=0.5, top_k=200):
    """ 非极大值抑制，去除多余的预测框

    Parameters
    ----------
    boxes: Tensor of shape `(n_priors, 4)`
        预测框，坐标形式为 `(xmin, ymin, xmax, ymax)`

    scores: Tensor of shape `(n_priors, )`
        某个类的每个先验框的置信度

    overlap_thresh: float
        IOU 阈值，大于阈值的部分预测框会被移除，值越小保留的框越多

    top_k: int
        保留的预测框个数上限

    Returns
    -------
    indexes: LongTensor of shape `(n, )`
        保留的边界框的索引
    """
    keep = []
    if boxes.numel() == 0:
        return torch.LongTensor(keep, device=scores.device)

    # 每个预测框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2-x1)*(y2-y1)

    # 对分数进行降序排序并截取前 top_k 个索引
    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[:top_k]

    while indexes.numel():
        i = indexes[0]
        keep.append(i)

        # 最后一个索引时直接退出循环
        if indexes.numel() == 1:
            break

        # 其他的预测框和当前预测框的交集
        right = x2[indexes].clamp(max=x2[i].item())
        left = x1[indexes].clamp(min=x1[i].item())
        bottom = y2[indexes].clamp(max=y2[i].item())
        top = y1[indexes].clamp(min=y1[i].item())
        inter = ((right-left)*(bottom-top)).clamp(min=0)

        # 计算 iou
        iou = inter/(area[i]+area[indexes]-inter)

        # 保留 iou 小于阈值的边界框，自己和自己的 iou 为 1
        indexes = indexes[iou < overlap_thresh]

    return torch.LongTensor(keep)


def draw(image: Union[ndarray, Image.Image], bbox: ndarray, label: ndarray, conf: ndarray = None) -> Image.Image:
    """ 在图像上绘制边界框和标签

    Parameters
    ----------
    image: `~np.ndarray` of shape `(H, W, 3)` or `~PIL.Image.Image`
        RGB 图像

    bbox: `~np.ndarray` of shape `(n_objects, 4)`
        边界框

    label: Iterable of shape `(n_objects, )`
        标签

    conf: Iterable of shape `(n_objects, )`
        置信度
    """
    bbox = bbox.astype(np.int)

    if isinstance(image, ndarray):
        image = Image.fromarray(image)  # type:Image.Image

    image_draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.truetype('./msyh.ttc', size=13)
    colors = cycle([
        '#0072BD', '#D95319', '#EDB120',
        '#7E2F8E', '#77AC30', '#4DBEEE',
        '#A2142F', '#7F7F7F', '#BCBD22', '#17BECF'
    ])

    for i in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[i]

        color = next(colors)

        # 绘制方框
        image_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 绘制标签
        y1_ = y1 if y1-23 < 0 else y1-23
        y2_ = y1 if y1_ < y1 else y1+23
        text = label[i] if conf is None else f'{label[i]} | {conf[i]}'
        l = font.getlength(text) + 3
        image_draw.rectangle([x1, y1_, x1+l, y2_],
                             fill=color+'75', outline=color+'DD')
        image_draw.text([x1+2, y1_+2], text=text,
                        font=font, embedded_color=color)

    return image
