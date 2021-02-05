import numpy as np
from base_layer import BaseLayer
import torch.nn as nn
import torch.nn.functional as F
from downsample_conv_2d import DownSampleConv2d

class Conv2dLayer(BaseLayer):
    def __init__(
            self,
            in_feature_map,
            out_feature_map,
            kernel,
            padding=0,
            down=False,
            resample_kernel=None,
            gain=1,
            use_wscale=True,
            lrmul=1):

        super(Conv2dLayer, self).__init__()

        self.down = down
        self.padding = padding

        # kh, kw, in_c, out_c
        w, runtime_coef = self.get_weight_and_runtime_coef(
            shape=[out_feature_map, in_feature_map, kernel, kernel],
            gain=gain,
            use_wscale=use_wscale,
            lrmul=lrmul)

        self.weight = nn.Parameter(w)
        self.runtime_coef = runtime_coef
        if self.down:
            self.downsample_conv_2d = DownSampleConv2d(resample_kernel=resample_kernel)

    def forward(self, x):
        if self.down:
            x = self.downsample_conv_2d(x, self.weight * self.runtime_coef)
        else:
            x = F.conv2d(x, self.weight * self.runtime_coef, padding=self.padding, stride=1)
        return x

    def get_weight_and_runtime_coef(self, shape, gain=1, use_wscale=True, lrmul=1):
        # [kernel, kernel, in_channel, out_channel] or [in, out]
        fan_in = np.prod(shape[1:])
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


