import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class L2Norm(nn.Module):
    """ L2 正则化 """

    def __init__(self, n_channels: int, scale):
        """
        Parameters
        ----------
        n_channels: int
            通道数

        scale: float
            l2正则化的缩放比
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
