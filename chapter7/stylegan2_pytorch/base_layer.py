import numpy as np
import torch.nn as nn
import torch
import pickle
from datetime import datetime
import os
import glob

class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.fmap_base = 16 << 10
        self.fmap_min = 1
        self.fmap_max = 512
        self.fmap_decay = 1

    def cliped_features(self, stage):
        before_clip_value = int(self.fmap_base / (2.0 ** (stage * self.fmap_decay)))
        cliped_value = np.clip(before_clip_value, self.fmap_min, self.fmap_max)
        return cliped_value

    def load_pkl(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file, encoding='latin1')

    def save_pkl(self, obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, model_path, dir_name):
        ret = False
        restore_model_list = sorted(glob.glob(os.path.join(model_path, dir_name, '*.pth')), reverse=True)
        if 0 < len(restore_model_list):
            restore_model_file = restore_model_list[0]
            self.load_state_dict(torch.load(restore_model_file))
            ret = True
        return ret

    def save(self, model_path, dir_name):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(model_path, dir_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, '{}.pth'.format(now)))

    @classmethod
    def save_model(cls, model_path, obj):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(model_path):
            os.makedirs(model_path, exist_ok=True)

        filename = os.path.join(model_path, '{}.pkl'.format(now))
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def restore_model(cls, model_path):
        ret = None
        restore_model_list = sorted(glob.glob(os.path.join(model_path, '*.pkl')), reverse=True)
        if len(restore_model_list) == 0:
            return ret
        restore_model_file = restore_model_list[0]
        with open(restore_model_file, 'rb') as file:
            ret = pickle.load(file, encoding='latin1')
        return ret

    @classmethod
    def print_model_parameters(cls, model, model_name):
        print('---- {} ----'.format(model_name))
        for index, key in enumerate(model.state_dict().keys()):
            print('index: {}, name: {}, shape: {}'.format(index, key, model.state_dict()[key].shape))

    @classmethod
    def set_model_parameter_requires_grad_all(cls, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
