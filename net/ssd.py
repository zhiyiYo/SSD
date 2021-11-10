# coding:utf-8
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.models.utils import load_state_dict_from_url

from .detector import Detector
from .prior_box import PriorBox


def vgg16(pretrained=False, batch_norm=False) -> nn.ModuleList:
    """ 创建 vgg16 模型

    Parameters
    ----------
    pretrained: bool
        是否使用预训练的模型

    batch_norm: bool
        是否在卷积层后面添加批归一化层
    """
    layers = []
    in_channels = 3
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'C', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=1)

            # 如果需要批归一化的操作就添加一个批归一化层
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(True)])
            else:
                layers.extend([conv, nn.ReLU(True)])

            in_channels = v

    # 将原始的 fc6、fc7 全连接层替换为卷积层
    layers.extend([
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # conv6 使用空洞卷积增加感受野
        nn.ReLU(True),
        nn.Conv2d(1024, 1024, 1),                        # conv7
        nn.ReLU(True)
    ])

    layers = nn.ModuleList(layers)

    # 使用预训练的 vgg16
    if pretrained:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/vgg16-397923af.pth", './model')
        state_dict = {k.replace('features.', ''): v for k,
                      v in state_dict.items()}
        layers.load_state_dict(state_dict, strict=False)

    return layers


class L2Norm(nn.Module):
    """ L2 标准化 """

    def __init__(self, n_channels: int, scale):
        """
        Parameters
        ----------
        n_channels: int
            通道数

        scale: float
            l2标准化的缩放比
        """
        super().__init__()
        self.gamma = scale
        self.eps = 1e-10
        self.n_channels = n_channels
        self.weight = nn.Parameter(Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x: Tensor):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x /= norm
        # 将 weight 的维度变为 [1, n_channels, 1, 1]
        x *= self.weight[None, ..., None, None]
        return x


class SSD(nn.Module):
    """ SSD 神经网络模型 """

    def __init__(self, n_classes: int, variance=(0.1, 0.2), top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        """
        Parameters
        ----------
        n_classes: int
            要预测的种类数，包括背景

        variance: Tuple[float, float]
            先验框的方差

        top_k: int
            每个类的边界框上限

        conf_thresh: float
            置信度阈值

        nms_thresh: float
            nms 中 IOU 阈值
        """
        super().__init__()

        if len(variance) != 2:
            raise ValueError("variance 只能有 2 元素")

        self.n_classes = n_classes
        self.priorbox_generator = PriorBox()
        self.prior = Tensor(self.priorbox_generator())
        self.detector = Detector(
            n_classes, variance, top_k, conf_thresh, nms_thresh)

        self.vgg16 = vgg16()
        self.l2norm = L2Norm(512, 20)
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),                        # conv8_2
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.Conv2d(512, 128, 1),                         # conv9_2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 128, 1),                         # conv10_2
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 128, 1),                         # conv11_2
            nn.Conv2d(128, 256, 3),
        ])
        # multi-box layers，用来回归和分类
        self.confs = nn.ModuleList([
            nn.Conv2d(512, n_classes*4, 3, padding=1),
            nn.Conv2d(1024, n_classes*6, 3, padding=1),
            nn.Conv2d(512, n_classes*6, 3, padding=1),
            nn.Conv2d(256, n_classes*6, 3, padding=1),
            nn.Conv2d(256, n_classes*4, 3, padding=1),
            nn.Conv2d(256, n_classes*4, 3, padding=1),
        ])
        self.locs = nn.ModuleList([
            nn.Conv2d(512, 4*4, 3, padding=1),
            nn.Conv2d(1024, 4*6, 3, padding=1),
            nn.Conv2d(512, 4*6, 3, padding=1),
            nn.Conv2d(256, 4*6, 3, padding=1),
            nn.Conv2d(256, 4*4, 3, padding=1),
            nn.Conv2d(256, 4*4, 3, padding=1),
        ])

    def forward(self, x: Tensor):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            图像数据

        Returns
        -------
        loc: Tensor of shape `(N, n_priors, 4)`
            偏移量

        conf: Tensor of shape `(N, n_priors, n_classes)`
            每个先验框中的类别置信度

        prior: Tensor of shape `(n_priors, 4)`
            先验框
        """
        loc = []
        conf = []
        sources = []

        # 批大小
        N = x.size(0)

        # 计算从 conv4_3 输出的特征图
        for layer in self.vgg16[:23]:
            x = layer(x)

        # 保存 conv4_3 输出的 l2 标准化结果
        sources.append(self.l2norm(x))

        # 计算 vgg16 后面几个卷积层的特征图
        for layer in self.vgg16[23:]:
            x = layer(x)

        # 保存 conv7 的输出的特征图
        sources.append(x)

        # 计算后面几个卷积层输出的特征图
        for i, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                sources.append(x)

        # 使用分类器和探测器进行预测并将通道变为最后一个维度方便堆叠
        for x, conf_layer, loc_layer in zip(sources, self.confs, self.locs):
            loc.append(loc_layer(x).permute(0, 2, 3, 1).contiguous())
            conf.append(conf_layer(x).permute(0, 2, 3, 1).contiguous())

        # 输出维度为 (batch_size, 8732, n_classes) 和 (batch_size, 8732, 4)
        conf = torch.cat([i.view(N, -1) for i in conf], dim=1)
        loc = torch.cat([i.view(N, -1) for i in loc], dim=1)

        return loc.view(N, -1, 4), conf.view(N, -1, self.n_classes),  self.prior

    def predict(self, x: Tensor):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            图像数据

        Returns
        -------
        out: Tensor of shape `(N, n_classes, top_k, 5)`
            检测结果，最后一个维度的前四个元素为边界框的坐标 `(xmin, ymin, xmax, ymax)`，最后一个元素为置信度
        """
        loc, conf, prior = self(x)
        out = self.detector(loc, F.softmax(conf, dim=1), prior)
        return out
