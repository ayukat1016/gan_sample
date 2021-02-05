import numpy as np
import torch.nn as nn
import torch
from dense_layer import DenseLayer
from fused_bias_activation import FusedBiasActivation
from base_layer import BaseLayer

from tensorboard_logger import TensorboardLogger

class MappingNetwork(BaseLayer):
    def __init__(self, dlaten_size, opt):
        super(MappingNetwork, self).__init__()
        self.mapping_layers = 8
        self.out_feature = 512

        resolution_log2 = int(np.log2(opt.resolution))
        self.num_layers = resolution_log2 * 2 - 2
        self.dense_layers = nn.ModuleDict()
        self.fused_bias_acts = nn.ModuleDict()
        for layer_idx in range(self.mapping_layers):
            self.dense_layers[str(layer_idx)] = DenseLayer(dlaten_size, self.out_feature, lmul=0.01)
            self.fused_bias_acts[str(layer_idx)] = FusedBiasActivation(dlaten_size, lrmul=0.01, act='LeakyRelu')

    def forward(self, z):
        x = self.normalize(z)
        for layer_idx in range(self.mapping_layers):
            x = self.dense_layers[str(layer_idx)](x)
            x = self.fused_bias_acts[str(layer_idx)](x)

        x = x.unsqueeze(1)
        x = x.repeat([1, self.num_layers, 1])
        return x

    def normalize(self, x):
        x_var = torch.mean(x**2, dim=1, keepdim=True)
        x_rstd = torch.rsqrt(x_var + 1e-8)
        normalized = x * x_rstd
        return normalized
