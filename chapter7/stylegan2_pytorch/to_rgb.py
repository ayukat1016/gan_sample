import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from modulate_conv import ModulateConv
from fused_bias_activation import FusedBiasActivation
from base_layer import BaseLayer

class ToRGB(BaseLayer):
    def __init__(self, x_channel, out_channel, kernel, style_dim, res):
        super(ToRGB, self).__init__()
        self.style_layer_index = res * 2 - 3

        self.modlate_conv2d = ModulateConv(
            x_channel=x_channel,
            feature_map=out_channel,
            style_in_dim=style_dim,
            style_out_dim=self.cliped_features(res - 1),
            # style_out_dim=style_dim,
            kernel=kernel,
            padding=0,
            is_demodulate=False)
        self.fused_bias_act = FusedBiasActivation(out_channel, act='Linear')

    def forward(self, x, style, before_image):
        # n, layer_num, out_ch(32, 8, 512) --> n, out_ch(32, 512)
        s = style[:, self.style_layer_index]

        # n, ch, h, w
        image = self.modlate_conv2d(x, s)
        image = self.fused_bias_act(image)
        if before_image is not None:
            image = image + before_image
        return image
