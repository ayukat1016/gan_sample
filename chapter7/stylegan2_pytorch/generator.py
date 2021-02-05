import numpy as np
import torch
import torch.nn as nn
import glob
import os
import random
from datetime import datetime
from mapping_network import MappingNetwork
from synthesis_network import SynthesisNetwork
from tensorboard_logger import TensorboardLogger
from base_layer import BaseLayer

class Generator(BaseLayer):
    def __init__(self, opt, truncation_psi=0.5, style_mixing_prob=0.9, dlatent_avg_beta=0.995):
        super(Generator, self).__init__()
        self.opt = opt
        self.dir_name = 'generator'
        self.mapping_nework = MappingNetwork(dlaten_size=opt.latent_dim, opt=opt)
        self.synthesis_nework = SynthesisNetwork(opt=opt)
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.dlatent_avg_beta = dlatent_avg_beta
        self.dlatent_avg = 0

    def forward(self, z):
        s1 = self.mapping_nework(z)
        s = s1
        if self.training:
            if self.style_mixing_prob is not None and random.random() < self.style_mixing_prob < 1:
                z2 = self.Tensor(np.random.normal(loc=0, scale=1, size=(s1.shape[0], self.opt.latent_dim)))
                s2 = self.mapping_nework(z2)
                mix_index = random.randint(0, self.mapping_nework.mapping_layers)
                s1_mix = s1[:, :mix_index, :]
                s2_mix = s2[:, mix_index:, :]
                s = torch.cat([s1_mix, s2_mix], 1)

            if self.dlatent_avg_beta is not None:
                batch_avg = torch.mean(s[:, 0], dim=0)
                self.dlatent_avg = batch_avg + (self.dlatent_avg - batch_avg) * self.dlatent_avg_beta
                # self.save_pkl(self.dlatent_avg, self.file_path)
        else:
            if self.truncation_psi is not None:
                # layer_idx = np.arange(self.mapping_nework.num_layers)[np.newaxis, :, np.newaxis]
                # layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
                # layer_psi *= self.truncation_psi
                # s = self.dlatent_avg + (s - self.dlatent_avg) * layer_psi
                s = self.dlatent_avg + (s - self.dlatent_avg) * self.truncation_psi

        x = self.synthesis_nework(s)
        return x, s

    @classmethod
    def apply_decay_parameters(cls, src_model, dest_model, decay):
        src_params = dict(src_model.named_parameters())
        dest_params = dict(dest_model.named_parameters())

        for key in dest_params.keys():
            dest_params[key].data.mul_(decay).add_(src_params[key].data, alpha=1 - decay)

        dest_model.dlatent_avg = src_model.dlatent_avg


