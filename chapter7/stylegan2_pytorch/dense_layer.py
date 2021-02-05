import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from fused_bias_activation import FusedBiasActivation
from base_layer import BaseLayer

class DenseLayer(BaseLayer):
    def __init__(self, in_channel, feature_map, gain=1, lmul=1):
        super(DenseLayer, self).__init__()

        w, runtime_coef = self.get_weight_and_runtime_coef(
            shape=[in_channel, feature_map],
            gain=gain,
            use_wscale=True,
            lrmul=lmul)
        self.weight = nn.Parameter(w)
        self.runtime_coef = runtime_coef

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        out = torch.matmul(x, self.weight * self.runtime_coef)
        return out

    def get_weight_and_runtime_coef(self, shape, gain=1, use_wscale=True, lrmul=1):
        # [kernel, kernel, in_channel, out_channel] or [in, out]
        fan_in = np.prod(shape[:-1])
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



