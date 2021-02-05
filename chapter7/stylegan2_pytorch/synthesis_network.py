import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from layer import Layer
from to_rgb import ToRGB
from block import Block
from upsample_2d import UpSample2d
from base_layer import BaseLayer
from tensorboard_logger import TensorboardLogger

class SynthesisNetwork(BaseLayer):
    def __init__(self, opt):
        super(SynthesisNetwork, self).__init__()
        self.opt = opt
        self.resolution = opt.resolution
        self.feature_maps = self.cliped_features(1)
        self.lmul = 1
        self.layers = []
        self.in_channel = self.cliped_features(1)
        self.out_rgb_channel = 3
        self.dlatent_size = 512

        self.resolution_log2 = int(np.log2(self.resolution))
        self.num_layers = self.resolution_log2 * 2 - 2

        # コンテンツデータの作成
        c = self.Tensor(np.random.normal(loc=0, scale=1, size=(1, 512, 4, 4)))
        # c = c.repeat([self.opt.batch_size, 1, 1, 1])
        # c.requires_grad = False
        self.const = nn.Parameter(c)

        self.layer = Layer(
            x_channel=self.in_channel,
            style_layer_index=0,
            style_in_dim=self.dlatent_size,
            style_out_dim=self.dlatent_size,
            feature_map=self.feature_maps,
            res=1)

        self.to_rgb = ToRGB(
            x_channel=self.in_channel,
            out_channel=self.out_rgb_channel,
            kernel=1,
            style_dim=self.dlatent_size,
            res=2)

        self.block_dict = nn.ModuleDict()
        self.to_rgb_dict = nn.ModuleDict()
        self.upsample_2d_dict = nn.ModuleDict()
        for res in range(3, self.resolution_log2 + 1):
            self.block_dict[str(res)] = Block(res, style_dim=self.dlatent_size)
            self.upsample_2d_dict[str(res)] = UpSample2d(res, resample_kernel=[1, 3, 3, 1])
            self.to_rgb_dict[str(res)] = ToRGB(
                self.cliped_features(res - 1),
                self.out_rgb_channel,
                kernel=1,
                style_dim=self.dlatent_size,
                res=res)

    def forward(self, style):
        image = None
        const_input = self.const.repeat([self.opt.batch_size, 1, 1, 1])

        # コンテンツデータとスタイルデータから出力を作成する
        x = self.layer(const_input, style)

        # imageデータに変換する
        image = self.to_rgb(x, style, image)

        for res in range(3, self.resolution_log2 + 1):
            x = self.block_dict[str(res)](x, style)
            upsampled_image = self.upsample_2d_dict[str(res)](image)
            image = self.to_rgb_dict[str(res)](x, style, upsampled_image)
        return image

