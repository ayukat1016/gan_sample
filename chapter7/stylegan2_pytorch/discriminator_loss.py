import numpy as np
import torch.nn as nn
import torch
from base_layer import BaseLayer

class DiscriminatorLoss(BaseLayer):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.softplus_fake = nn.Softplus()
        self.softplus_real = nn.Softplus()

    def forward(self, fake_scores_out, real_scores_out):
        loss = torch.mean(self.softplus_fake(fake_scores_out)) + torch.mean(self.softplus_real(-real_scores_out))
        return loss

