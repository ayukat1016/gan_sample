import numpy as np
import torch.nn as nn
import torch
from base_layer import BaseLayer
import os
import pickle

class GeneratorLossPathReg(BaseLayer):
    def __init__(self, pl_decay=0.01, pl_weight=2.0, opt=None):
        super(GeneratorLossPathReg, self).__init__()
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean_var = self.Tensor(np.zeros(1,))
        self.reg_interval = 4 if opt is None else opt.g_reg_interval
        self.opt = opt
        self.file_path = os.path.join(opt.cache_path, 'pl_mean_var.pkl')
        if opt is not None and opt.is_restore_model:
            if os.path.isfile(self.file_path):
                self.pl_mean_var = self.Tensor(self.load_pkl(self.file_path))

    def forward(self, fake_images_out, fake_dlatents_out):
        # Compute |J*y|.
        pl_noise = self.Tensor(np.random.normal(0, 1, fake_images_out.shape)) / np.sqrt(np.prod(fake_images_out.shape[2:]))
        f_img_out_pl_n = torch.sum(fake_images_out * pl_noise)
        pl_grads = torch.autograd.grad(outputs=f_img_out_pl_n, inputs=fake_dlatents_out, create_graph=True)[0]
        pl_grads_sum_mean = pl_grads.pow(2).sum(dim=2).mean(dim=1)
        pl_length = torch.sqrt(pl_grads_sum_mean)

        # Track exponential moving average of |J*y|.
        pl_mean = self.pl_mean_var + self.pl_decay * (pl_length.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach()
        self.save_pkl(self.pl_mean_var, self.file_path)

        # Calculate (|J*y|-a)^2.
        pl_penalty = (pl_length - pl_mean).pow(2).mean()
        reg = pl_penalty * self.pl_weight * self.reg_interval
        return reg, pl_length.mean()

