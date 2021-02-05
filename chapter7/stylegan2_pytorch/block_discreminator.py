import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from layer import Layer
from to_rgb import ToRGB
from base_layer import BaseLayer
from conv2d_layer import Conv2dLayer
from fused_bias_activation import FusedBiasActivation

class Block(BaseLayer):
    def __init__(self, res):
        super(Block, self).__init__()

        self.conv2d_layer = Conv2dLayer(
            in_feature_map=self.cliped_features(res - 1),
            out_feature_map=self.cliped_features(res - 1),
            kernel=3,
            padding=1)
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(res - 1), act='LeakyRelu')

        self.conv2d_layer_down1 = Conv2dLayer(
            in_feature_map=self.cliped_features(res - 1),
            out_feature_map=self.cliped_features(res - 2),
            kernel=3,
            padding=1,
            down=True,
            resample_kernel=[1, 3, 3, 1])
        self.fused_bias_act1 = FusedBiasActivation(channel=self.cliped_features(res - 2), act='LeakyRelu')

        self.conv2d_layer_down2 = Conv2dLayer(
            in_feature_map=self.cliped_features(res - 1),
            out_feature_map=self.cliped_features(res - 2),
            kernel=1,
            down=True,
            resample_kernel=[1, 3, 3, 1])
        # self.fused_bias_act2 = FusedBiasActivation(self.cliped_features(res - 2))

    def forward(self, x):
        t = x
        x = self.conv2d_layer(x)
        x = self.fused_bias_act(x)

        x = self.conv2d_layer_down1(x)
        x = self.fused_bias_act1(x)

        t = self.conv2d_layer_down2(t)
        x = (x + t) * (1 / np.sqrt(2))
        return x

