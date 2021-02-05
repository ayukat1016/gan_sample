import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from layer import Layer
from to_rgb import ToRGB
from base_layer import BaseLayer

class Block(BaseLayer):
    def __init__(self, res, style_dim):
        super(Block, self).__init__()
        self.x_channel = self.cliped_features(res - 2)
        self.feature_map = self.cliped_features(res - 1)

        # 1, 3, 5, 7, 9, 11
        layer_index_upsample = res * 2 - 5
        self.layer_upsample = Layer(
            x_channel=self.x_channel,
            style_layer_index=layer_index_upsample,
            style_in_dim=style_dim,
            style_out_dim=style_dim,
            feature_map=self.feature_map,
            res=res,
            is_up=True)

        # 2, 4, 6, 8, 10, 12
        layer_index = res * 2 - 4
        self.layer = Layer(
            x_channel=self.feature_map,
            style_layer_index=layer_index,
            style_in_dim=style_dim,
            style_out_dim=self.cliped_features(res - 1),
            feature_map=self.feature_map,
            res=res)

    def forward(self, x, style):
        x = self.layer_upsample(x, style)
        x = self.layer(x, style)
        return x

