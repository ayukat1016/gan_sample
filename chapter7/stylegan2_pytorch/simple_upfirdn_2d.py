import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from fused_bias_activation import FusedBiasActivation
from torch.nn import ConvTranspose2d

class SimpleUpfirdn2d(nn.Module):
    def __init__(self, resample_kernel=None, factor=2, gain=1, padding=0):
        super(SimpleUpfirdn2d, self).__init__()
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.resample_kernel = resample_kernel
        self.factor = factor
        self.gain = gain
        self.padding = padding

        # self.resample_kernel = self.resample_kernel[np.newaxis, np.newaxis, ::-1, ::-1]
        # self.weight = nn.Parameter(self.Tensor(self.resample_kernel[np.newaxis, np.newaxis, ::, ::]))
        self.weight = self.Tensor(self.resample_kernel[np.newaxis, np.newaxis, ::, ::])
        self.weight = self.weight.flip(2)
        self.weight = self.weight.flip(3)

    def forward(self, x_, resample_kernel, up=1, down=1, pad0=0, pad1=0):
        # x_ = x

        # 1, n*in_c, h, w --> n*in_c, h, w, 1
        x_ = x_.view((-1, x_.shape[2], x_.shape[3], 1))

        upx = up
        upy = up
        downx = down
        downy = down
        padx0 = pad0
        padx1 = pad1
        pady0 = pad0
        pady1 = pad1

        in_height = x_.shape[1]
        in_width = x_.shape[2]
        minor_dim = x_.shape[3]
        kernel_h, kernel_w = resample_kernel.shape

        # upsample(ゼロを入れる)
        #  n*in_c, h, w, 1 --> n*in_c, h, 1, w, 1, 1
        x_ = x_.reshape((-1, 1, in_height, 1,  in_width, 1, minor_dim))
        # x_ = x_.reshape((-1, in_height, 1,  in_width, 1,  minor_dim))
        pad00 = [0, 0]
        pad01 = [0, upy - 1]
        pad02 = [0, 0]
        pad03 = [0, upx - 1]
        pad04 = [0, 0]
        pad05 = [0, 0]
        pad06 = [0, 0]
        # pad_all = batch_in_ch_pad + in_height_pad + in_h_in_w_pad + in_width_pad + in_w_minor_dim_pad + minor_dim_pad
        pad_all = pad00 + pad01 + pad02 + pad03 + pad04 + pad05 + pad06
        x_ = F.pad(x_, pad_all)
        x_ = x_.reshape((-1, in_height * upy, in_width * upx, minor_dim))

        # crop
        in_height_pad = [max(pady0, 0), max(pady1, 0)]
        in_width_pad = [max(padx0, 0), max(padx1, 0)]
        pad_crop = pad00 + in_height_pad + in_width_pad + pad06
        x_ = F.pad(x_, pad_crop)
        x_ = x_[:, max(-pady0, 0):x_.shape[1] - max(-pady1, 0), max(-padx0, 0):x_.shape[2] - max(-padx1, 0), :]

        # convolution
        # n*in_c, h, w, 1 --> n*in_c, 1, h, w
        x_ = x_.permute(0, 3, 1, 2)

        # n*in_c, 1, new_h, new_w
        x_ = x_.reshape((-1, 1, in_height * upy + pady0 + pady1, in_width * upx + padx0 + padx1))
        x_ = F.conv2d(x_, self.weight, padding=self.padding, stride=1)
        height = in_height * upy + pady0 + pady1 - kernel_h + 1
        width = in_width * upx + padx0 + padx1 - kernel_w + 1
        x_ = x_.reshape((-1, minor_dim, height, width))
        x_ = x_.permute(1, 0, 2, 3)

        # down sample
        x_ = x_[:, :, ::downy, ::downx]
        return x_

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

