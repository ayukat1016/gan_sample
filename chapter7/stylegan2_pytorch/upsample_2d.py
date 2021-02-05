import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from fused_bias_activation import FusedBiasActivation
from torch.nn import ConvTranspose2d
from simple_upfirdn_2d import SimpleUpfirdn2d
from base_layer import BaseLayer

class UpSample2d(BaseLayer):
    def __init__(self, res, resample_kernel=None, factor=2, gain=1):
        super(UpSample2d, self).__init__()
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

    def forward(self, image):
        # image:[n, 3, h, w]
        batch_size, channel, height, width = image.shape
        p = self.resample_kernel.shape[0] - self.factor
        pad0 = (p + 1) // 2 + self.factor - 1
        pad1 = p // 2
        upsampled_image = self.simple_upfirdn_2d(image, self.resample_kernel, up=self.factor, pad0=pad0, pad1=pad1)
        upsampled_image = upsampled_image.view((-1, channel, upsampled_image.shape[2], upsampled_image.shape[3]))
        return upsampled_image

