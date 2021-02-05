import numpy as np
try:
    import accimage
except ImportError:
    accimage = None
import torch

class TranformDynamicRange(object):
    def __init__(self, drange_in, drange_out):
        self.drange_in = drange_in
        self.drange_out = drange_out
        self.scale = (np.float32(self.drange_out[1]) - np.float32(self.drange_out[0])) / \
                     (np.float32(self.drange_in[1]) - np.float32(self.drange_in[0]))
        self.bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * self.scale)

    def __call__(self, x):
        # transform.ToTensor()が事前に呼ばれている前提
        # 0〜1の範囲になっているので、255をかけてもとに戻す
        x = x * 255
        x = x * self.scale + self.bias
        # x = self.fade_lod(x)
        return x

    @staticmethod
    def fade_lod(x, lod):
        shape = x.shape
        y = x.reshape((-1, shape[1], shape[2]//2, 2, shape[3]//2, 2))
        y = y.mean(axis=[3, 5], keepdims=True)
        y = y.repeat([1, 1, 1, 2, 1, 2])
        y = y.reshape((-1, shape[1], shape[2], shape[3]))
        # x = tflib.lerp(x, y, self.lod - tf.floor(self.lod))
        x = x + (y - x) * (lod - np.floor(lod))
        return x

    @staticmethod
    def upscale_lod(x, lod):
        shape = x.shape
        factor = int(2 ** np.floor(lod))
        x = x.reshape((-1, shape[1], shape[2], 1, shape[3], 1))
        x = x.repeat([1, 1, 1, factor, 1, factor])
        x = x.reshape((-1, shape[1], shape[2] * factor, shape[3] * factor))
        return x



