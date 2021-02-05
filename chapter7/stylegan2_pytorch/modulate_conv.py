import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from dense_layer import DenseLayer
from fused_bias_activation import FusedBiasActivation
from upsample_conv_2d import UpSampleConv2d
from base_layer import BaseLayer
from tensorboard_logger import TensorboardLogger

class ModulateConv(BaseLayer):
    def __init__(self,
                 x_channel,
                 feature_map,
                 style_in_dim,
                 style_out_dim,
                 kernel=3,
                 padding=0,
                 is_demodulate=True,
                 is_up=False):

        super(ModulateConv, self).__init__()
        self.feature_map = feature_map
        self.padding = padding
        self.stride = 1
        self.is_demodulate = is_demodulate
        self.is_up = is_up

        # out_c, in_c, kh, kw
        w, runtime_coef = self.get_weight_and_runtime_coef(
            shape=[self.feature_map, x_channel, kernel, kernel],
            gain=1,
            use_wscale=True,
            lrmul=1)
        self.weight = nn.Parameter(w)
        self.runtime_coef = runtime_coef

        # self.dense_layer = DenseLayer(in_channel=style_in_dim, feature_map=style_out_dim)
        self.dense_layer = DenseLayer(in_channel=style_in_dim, feature_map=x_channel)

        # self.fused_bias_act = FusedBiasActivation(channel=style_out_dim, act='Linear')
        self.fused_bias_act = FusedBiasActivation(channel=x_channel, act='Linear')

        if self.is_up:
            self.upsample_conv_2d = UpSampleConv2d()

    def forward(self, x, style):
        # weight: out_c, in_c, kh, kw
        # style:  batch, f_map
        # w_m:    batch, out_c, in_c, kh, kw
        #            16,   512,  512,  3,  3
        w_m = self.modulate(self.weight * self.runtime_coef, style)

        if self.is_demodulate:
            # batch, out_c, in_c, kh, kw
            #    16,   512,  512,  3,  3
            w_ = self.demodulate(w_m)
        else:
            w_ = w_m

        batch, out_c, in_c, kh, kw = w_.shape
        # 16*512, 512, 3, 3
        w = w_.view(batch * out_c, in_c, kh, kw)

        x_batch, x_c, x_h, x_w = x.shape
        # 1, 16*512, 4, 4
        x = x.view(1, x_batch * x_c, x_h, x_w)

        if self.is_up:
            # 1, batch*out_c, 4, 4 --> 1, batch*out_c, 8, 8
            x = self.upsample_conv_2d(x, w)
        else:
            # x:             1,   batch*x_c, x_h, x_w
            # w:   batch*out_c,        in_c,  kh,  kw
            # out:           1, batch*out_c, x_h, x_w
            # x:             1,   16*512, 4, 4
            # w:        16*512,      512, 3, 3
            # out:           1,   16*512, 4, 4
            x = F.conv2d(x, w, padding=self.padding, stride=self.stride, groups=batch)

        x = x.view(x_batch, self.feature_map, x.shape[2], x.shape[3])
        return x

    def get_weight_and_runtime_coef(self, shape, gain=1, use_wscale=True, lrmul=1):
        # [kernel, kernel, in_channel, out_channel] or [in, out]
        fan_in = np.prod(shape[1:])
        he_std = gain / np.sqrt(fan_in) # He init

        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            runtime_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            runtime_coef = lrmul

        # Create variable.
        w = self.Tensor(np.random.normal(loc=0, scale=init_std, size=shape))
        return w, runtime_coef

    def modulate(self, weight, style_data):
        out_c, in_c, kh, kw = weight.shape

        # out_c, in_c, kh, kw --> 1, out_c, in_c, kh, kw
        #   512,  512,  3,  3 --> 1,   512,  512,  3,  3
        w = weight.view(1, out_c, in_c, kh, kw)

        batch, f_map = style_data.shape
        s_d = self.dense_layer(style_data)
        s_b = self.fused_bias_act(s_d) + 1

        # batch, f_map --> batch, 1, in_c, 1, 1
        #    16,   512 -->    16, 1,  512, 1, 1
        s = s_b.view(batch, 1, in_c, 1, 1)

        # w:                 1, out_c, in_c, kh, kw
        # s:             batch,     1, in_c,  1,  1
        # weight_scaled: batch, out_c, in_c, kh, kw

        # w:                 1,   512,  512,  3,  3
        # s:                16,     1,  512,  1,  1
        # weight_scaled:    16,   512,  512,  3,  3
        weight_scaled = w * s
        return weight_scaled

    def demodulate(self, weight):
        # weight: batch, out_c, in_c, kh, kw
        # weight:    16,   512,  512,  3,  3
        batch, out_c, in_c, kh, kw = weight.shape

        # batch, out_c
        #    16,   512
        ww_sum = weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8

        # batch, out_c, 1, 1, 1
        #    16,   512, 1, 1, 1
        r_dev = torch.rsqrt(ww_sum).view(batch, out_c, 1, 1, 1)
        ret = weight * r_dev
        return ret

    def cliped_features(self, stage):
        before_clip_value = int(self.fmap_base / (2.0 ** (stage * self.fmap_decay)))
        cliped_value = np.clip(before_clip_value, self.fmap_min, self.fmap_max)
        return cliped_value

