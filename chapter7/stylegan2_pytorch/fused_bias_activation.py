import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from base_layer import BaseLayer

class FusedBiasActivation(BaseLayer):
    def __init__(self, channel, act='Linear', alpha=None, gain=None, lrmul=1,  channel_axis=1):
        super(FusedBiasActivation, self).__init__()
        self.lrmul = lrmul
        self.bias = nn.Parameter(self.Tensor(np.zeros(channel)))
        # self.bias = nn.Parameter(torch.zeros(channel))
        # torch.nn.init.zeros_(self.bias.data)

        self.actvation = None
        if act == 'Linear':
            # self.actvation = nn.Linear(channel, channel)
            # self.actvation = lambda x, **kwargs : x
            self.actvation = self.linear
            self.gain = 1.0 if gain is None else gain
        else:
            self.alpha = 0.2 if alpha is None else alpha
            self.actvation = nn.LeakyReLU(negative_slope=self.alpha)
            self.gain = np.sqrt(2) if gain is None else gain

        self.channel_axis = channel_axis

    def forward(self, x):
        # 1, channel, 1, 1
        bias_shape = [-1 if i == self.channel_axis else 1 for i in range(len(x.shape))]
        b = self.bias.view(tuple(bias_shape)) * self.lrmul

        # n, ch, h, w + 1, ch, 1, 1
        x = x + b
        x = self.actvation(x)
        if self.gain != 1:
            x = x * self.gain
        return x

    def linear(self, x):
        return x