import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from modulate_conv import ModulateConv
from fused_bias_activation import FusedBiasActivation
from base_layer import BaseLayer
from tensorboard_logger import TensorboardLogger

class Layer(BaseLayer):
    def __init__(self, x_channel, style_layer_index, style_in_dim, style_out_dim, feature_map, res, kernel=3, is_up=False):
        super(Layer, self).__init__()
        self.noise_strength = nn.Parameter(self.Tensor(np.zeros(1)))
        # self.noise_strength.requires_grad = True
        self.style_layer_index = style_layer_index

        self.modlate_conv2d = ModulateConv(
            x_channel=x_channel,
            feature_map=feature_map,
            style_in_dim=style_in_dim,
            # style_out_dim=self.cliped_features(res - 1),
            style_out_dim=style_out_dim,
            kernel=kernel,
            padding=1,  # padding same
            is_demodulate=True,
            is_up=is_up)
        self.fused_bias_act = FusedBiasActivation(feature_map, act='LeakyReLU')

    def forward(self, x, style):
        s = style[:, self.style_layer_index]
        x = self.modlate_conv2d(x, s)
        batch_size, channel, height, width = x.shape
        noise = self.Tensor(np.random.normal(loc=0, scale=1, size=(batch_size, 1, height, width)))
        noise = noise * self.noise_strength
        x = x + noise
        x = self.fused_bias_act(x)
        return x


