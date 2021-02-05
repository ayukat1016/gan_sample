import numpy as np
import torch.nn as nn
import torch
from base_layer import BaseLayer

class GeneratorLoss(BaseLayer):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, fake_scores_out):
        loss = torch.mean(self.softplus(-fake_scores_out))
        return loss
