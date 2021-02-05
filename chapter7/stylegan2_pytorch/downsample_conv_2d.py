import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from fused_bias_activation import FusedBiasActivation
from torch.nn import ConvTranspose2d
from simple_upfirdn_2d import SimpleUpfirdn2d
from base_layer import BaseLayer

class DownSampleConv2d(BaseLayer):
    def __init__(self, resample_kernel=None, factor=2, gain=1):
        super(DownSampleConv2d, self).__init__()
        self.padding = 0
        self.stride = (factor, factor)
        self.factor = factor

        self.resample_kernel = resample_kernel
        if resample_kernel is None:
            self.resample_kernel = [1, 3, 3, 1]
        self.resample_kernel = np.asarray(self.resample_kernel, dtype=np.float32)
        if self.resample_kernel.ndim == 1:
            self.resample_kernel = np.outer(self.resample_kernel, self.resample_kernel)
        self.resample_kernel /= np.sum(self.resample_kernel)
        self.resample_kernel = self.resample_kernel * gain

        self.simple_upfirdn_2d = SimpleUpfirdn2d(self.resample_kernel)

        # self.weight = nn.Parameter(self.Tensor(self.resample_kernel[::-1, ::-1, np.newaxis, np.newaxis]))
        # self.weight = nn.Parameter(self.Tensor(self.resample_kernel[np.newaxis, np.newaxis, ::, ::]))

    def forward(self, x, w):
        # x:[1, n*channel, h, w]
        # w:[out_c, in_c, kh, kw]
        out_channel, in_channel, kernel_height, kernel_width = w.shape

        p = (self.resample_kernel.shape[0] - self.factor) + (kernel_width - 1)
        pad0 = (p + 1) // 2
        pad1 = p // 2
        x_channel = x.shape[1]
        x = self.simple_upfirdn_2d(x, self.resample_kernel, pad0=pad0, pad1=pad1)
        x = x.view((-1, x_channel, x.shape[2], x.shape[3]))
        x = F.conv2d(x, w, padding=self.padding, stride=self.stride)
        return x


