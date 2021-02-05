import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from modulate_conv import ModulateConv
from fused_bias_activation import FusedBiasActivation
from base_layer import BaseLayer
from conv2d_layer import Conv2dLayer
from fused_bias_activation import FusedBiasActivation

class FromRGB(BaseLayer):
    def __init__(self, res):
        super(FromRGB, self).__init__()
        self.conv2d_layer = Conv2dLayer(in_feature_map=3, out_feature_map=self.cliped_features(res - 1), kernel=1)
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(res - 1), act='LeakyRelu')

    def forward(self, x, image):
        t = self.conv2d_layer(image)
        t = self.fused_bias_act(t)
        return t if x is None else x + t



