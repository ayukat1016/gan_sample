import os
import torch.nn as nn
import torch
import numpy as np
import glob
from base_layer import BaseLayer
from from_rgb import FromRGB
from block_discreminator import Block
from minibatch_stddev_layer import MiniBatchStddevLayer
from conv2d_layer import Conv2dLayer
from fused_bias_activation import FusedBiasActivation
from dense_layer import DenseLayer

class Discriminator(BaseLayer):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.dir_name = 'discriminator'
        self.resolution = opt.resolution
        self.resolution_log2 = int(np.log2(self.resolution))
        self.mbstd_group_size = 4
        self.mbstd_num_features = 1

        self.d_block_dict = nn.ModuleDict()
        for res in range(self.resolution_log2, 2, -1):
            if res == self.resolution_log2:
                self.fromrgb = FromRGB(res)
            self.d_block_dict[str(res)] = Block(res)

        self.minibatch_stddev_layer = MiniBatchStddevLayer(self.mbstd_group_size, self.mbstd_num_features)

        # x: [32, 513, 4, 4]
        in_feature_map = self.cliped_features(1) + 1
        self.conv2d_layer = Conv2dLayer(in_feature_map=in_feature_map, out_feature_map=self.cliped_features(1), kernel=3, padding=1)
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(1), act='LeakyRelu')

        in_channel = self.cliped_features(0) * 4**2
        self.dense_layer1 = DenseLayer(in_channel=in_channel, feature_map=self.cliped_features(0))
        self.fused_bias_act1 = FusedBiasActivation(channel=self.cliped_features(0), act='LeakyRelu')

        self.dense_layer2 = DenseLayer(in_channel=self.cliped_features(0), feature_map=1)
        self.fused_bias_act2 = FusedBiasActivation(channel=1)

    def forward(self, image):
        x = None
        for res in range(self.resolution_log2, 2, -1):
            if res == self.resolution_log2:
                x = self.fromrgb(x, image)
            x = self.d_block_dict[str(res)](x)

        if self.mbstd_group_size > 1:
            x = self.minibatch_stddev_layer(x)

        x = self.conv2d_layer(x)
        x = self.fused_bias_act(x)

        x = self.dense_layer1(x)
        x = self.fused_bias_act1(x)

        x = self.dense_layer2(x)
        x = self.fused_bias_act2(x)

        return x

