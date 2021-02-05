import numpy as np
import torch.nn as nn
import torch
from base_layer import BaseLayer

class DiscriminatorLossR1(BaseLayer):
    def __init__(self, gamma=10.0, reg_interval=16):
        super(DiscriminatorLossR1, self).__init__()
        self.gamma = gamma
        self.reg_interval = reg_interval

    def forward(self, real_scores_out, reals):
        real_grads = torch.autograd.grad(outputs=torch.sum(real_scores_out), inputs=reals, create_graph=True)[0]
        gradient_penalty = torch.sum(real_grads**2, dim=[1, 2, 3])
        reg = (gradient_penalty * self.gamma * 0.5 * self.reg_interval).mean()
        return reg

