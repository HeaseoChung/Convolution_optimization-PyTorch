import torch.nn as nn
from torch.nn import functional as F


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3 // 2)
            )
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res * self.res_scale


class ResBlock_ds_conv(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock_ds_conv, self).__init__()
        m = []
        for i in range(2):
            m.append(
                depthwise_separable_conv(
                    n_feats, n_feats, kernel_size=3, bias=True, padding=3 // 2
                )
            )
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res * self.res_scale


class EDSR(nn.Module):
    def __init__(
        self, scale_factor=2, num_channels=3, num_feats=64, num_blocks=16, res_scale=1.0
    ):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=3 // 2)
        body = [ResBlock_ds_conv(num_feats, res_scale) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(
                num_feats,
                num_feats * (scale_factor ** 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(True),
            nn.Conv2d(num_feats, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
