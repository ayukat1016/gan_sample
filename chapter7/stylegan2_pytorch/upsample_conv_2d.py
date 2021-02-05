import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from fused_bias_activation import FusedBiasActivation
from torch.nn import ConvTranspose2d
from simple_upfirdn_2d import SimpleUpfirdn2d
from base_layer import BaseLayer
from tensorboard_logger import TensorboardLogger

class UpSampleConv2d(BaseLayer):
    def __init__(self, resample_kernel=None, factor=2, gain=1):
        super(UpSampleConv2d, self).__init__()
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
        self.resample_kernel = self.resample_kernel * (gain * (self.factor ** 2))

        self.simple_upfirdn_2d = SimpleUpfirdn2d(self.resample_kernel)

        # self.weight = nn.Parameter(self.Tensor(self.resample_kernel[::-1, ::-1, np.newaxis, np.newaxis]))
        # self.weight = nn.Parameter(self.Tensor(self.resample_kernel[np.newaxis, np.newaxis, ::, ::]))

    def forward(self, x, w):
        # x:[1, n*channel, h, w]
        # w:[batch*out_c, in_c, kh, kw]
        out_channel, in_channel, kernel_height, kernel_width = w.shape
        num_groups = x.shape[1] // in_channel

        w = w.view(num_groups, -1, in_channel, kernel_height, kernel_width)
        w = w.transpose(1, 2).reshape(num_groups*in_channel, -1, kernel_height, kernel_width)
        x = F.conv_transpose2d(x, w, padding=self.padding, stride=self.stride, groups=num_groups)

        p = (self.resample_kernel.shape[0] - self.factor) - (kernel_width - 1)
        pad0 = (p + 1) // 2 + self.factor - 1
        pad1 = p // 2 + 1
        x = self.simple_upfirdn_2d(x, self.resample_kernel, pad0=pad0, pad1=pad1)
        return x

    def forward_org(self, x, w_):
        # x:[1, n*channel, h, w]
        # w:[out_c, in_c, kh, kw]
        out_channel, in_channel, kernel_height, kernel_width = w_.shape
        num_groups = x.shape[1] // in_channel

        # ouu_c*n, in_c, kh, kw --> kh, kw, in_c, out_c*n,
        w1 = w_.permute(2, 3, 1, 0)

        # kh, kw, in_c, out_c*n --> kh, kw, in_c, n, -1(out_c)
        w2 = w1.view(kernel_height, kernel_width, in_channel, num_groups, -1)

        # w = w[::-1, ::-1]
        w3 = self.flip(w2, 0)
        w4 = self.flip(w3, 1)

        # kh, kw, in_c, num_groups, -1(out_c) --> in_c, num_groups, -1(out_c), kh, kw
        w5 = w4.permute(2, 3, 4, 0, 1)

        # in_c, num_groups, -1(out_c), kh, kw --> in_c * num_groups, -1(out_c) ,kh, kw
        w = w5.contiguous().view(in_channel * num_groups, -1, kernel_height, kernel_width)
        x = F.conv_transpose2d(x, w, padding=self.padding, stride=self.stride, groups=num_groups)
        return x

        p = (self.resample_kernel.shape[0] - self.factor) - (kernel_width - 1)
        pad0 = (p + 1) // 2 + self.factor - 1
        pad1 = p // 2 + 1
        x = self.simple_upfirdn_2d(x, self.resample_kernel, pad0=pad0, pad1=pad1)

        return x

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

