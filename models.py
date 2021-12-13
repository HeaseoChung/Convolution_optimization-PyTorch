import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp

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
    def __init__(self, n_feats, res_scale=1.0, conv_type="standard"):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if conv_type == "standard":
                m.append(
                    nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3//2)
                )
            elif conv_type == "depthwise_separable":
                depthwise_separable_conv(
                    n_feats, n_feats, kernel_size=3, bias=True, padding=3//2
                )
            else:
                raise ValueError("Please select right block type")

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
        self, scale=2, num_channels=3, num_feats=64, num_blocks=16, res_scale=1.0, conv_type="standard"
    ):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=3 // 2)

        body = [ResBlock(num_feats, res_scale, conv_type) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(
                num_feats,
                num_feats * (scale ** 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale),
            nn.ReLU(True),
            nn.Conv2d(num_feats, num_channels, kernel_size=3, stride=1, padding=1),
        )
     
    @amp.autocast()
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
