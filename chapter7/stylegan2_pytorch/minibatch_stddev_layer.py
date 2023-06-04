from base_layer import BaseLayer
import torch
import numpy as np
import math

class MiniBatchStddevLayer(BaseLayer):
    def __init__(self, mbstd_group_size, mbstd_num_features):
        super(MiniBatchStddevLayer, self).__init__()
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features

    def forward(self, x):
        # x: [32, 512, 4, 4]
        group_size = min(self.mbstd_group_size, x.shape[0])
        batch_size, channel, height, width = x.shape

        # y: [4, 8, 1, 512, 4, 4]
        y = x.view((group_size, -1, self.mbstd_num_features, channel//self.mbstd_num_features, height, width))

        # y.mean: [1, 8, 1, 512, 4, 4]
        # y: [4, 8, 1, 512, 4, 4]
        y = y - y.mean(dim=0, keepdims=True)
        y = y * y

        # y: [8, 1 512, 4, 4]
        y = y.mean(dim=0)
        y = torch.sqrt(y + 1e-8)

        # y: [8, 1, 1, 1, 1]
        y = y.mean(dim=[2, 3, 4], keepdim=True)

        # y: [8, 1, 1, 1]
        y = y.mean(dim=2)

        # y: [8*group_size, 1, 4, 4]
        y = y.repeat([group_size, 1, height, width])

        # x: [32, 513, 4, 4]
        x = torch.cat([x, y], dim=1)
        return x
