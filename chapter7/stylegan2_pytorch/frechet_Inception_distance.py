import numpy as np
import os
import pickle
from scipy import linalg
from distutils.util import strtobool
from inception import InceptionV3
import torchvision.transforms as transforms
from transform import TranformDynamicRange
import torch
from dataset import TrainDataset
from base_layer import BaseLayer
import argparse
from generator import Generator
from tensorboard_logger import TensorboardLogger
from datasource import get_dataloader


class FrechetInceptionDistance(BaseLayer):
    def __init__(self, generator, dataloader, opt):
        super(FrechetInceptionDistance, self).__init__()
        self.generator = generator
        # self.generator.eval()
        self.batch_size = opt.batch_size
        self.latent_dim = opt.latent_dim
        self.data_path = opt.data_path
        self.dataloader = dataloader
        self.all_data_num = len(self.dataloader.dataset)
        self.inception = InceptionV3([3], normalize_input=False)
        self.cache_dir = opt.cache_path
        self.cache_filename = '{}_real_image_mean_cov.pkl'.format(self.data_path.split('/')[-1])
        self.cache_file_path = os.path.join(self.cache_dir, self.cache_filename)

    def get_score(self):
        print('---- calculate fid score ----')
        if os.path.isfile(self.cache_file_path):
            print('reading real mean and cov from cache file')
            real_features, real_mean, real_cov = self.load_pkl(self.cache_file_path)
        else:
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)

            feature_list = []
            print('extract features from real image')
            for index, imgs in enumerate(self.dataloader):
                feature = self.inception(imgs)[0].view(imgs.shape[0], -1)
                feature_list.append(feature)
                if index % 16 == 0:
                    print('{} was done'.format(index))

            real_features = torch.cat(feature_list, 0).to('cpu').detach().numpy()
            real_mean = np.mean(real_features, axis=0)
            print('calculating real images mean was done')

            real_cov = np.cov(real_features, rowvar=False)
            print('calculating real images coveriance was done')
            self.save_pkl((real_features, real_mean, real_cov), self.cache_file_path)

        batch_loop_num = len(self.dataloader)
        fake_feature_list = []
        print('extract features from fake image')
        for index in range(batch_loop_num):
            latent = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.batch_size, self.latent_dim)))
            fake_imgs, _ = self.generator(latent)
            fake_imgs = fake_imgs.to('cpu').detach()
            fake_feature = self.inception(fake_imgs)[0].view(fake_imgs.shape[0], -1)
            fake_feature_list.append(fake_feature)
            if index % 16 == 0:
                print('{} was done'.format(index))

        fake_features = torch.cat(fake_feature_list, 0).to('cpu').detach().numpy()
        if self.all_data_num < fake_features.shape[0]:
            fake_features = fake_features[:self.all_data_num]

        TensorboardLogger.writer.add_histogram('{}/fid/real_features'.format(TensorboardLogger.now), real_features, TensorboardLogger.global_step)
        TensorboardLogger.writer.add_histogram('{}/fid/fake_features'.format(TensorboardLogger.now), fake_features, TensorboardLogger.global_step)

        fake_mean = np.mean(fake_features, axis=0)
        print('calculating fake images mean was done')

        fake_cov = np.cov(fake_features, rowvar=False)
        print('calculating fake images coveriance was done')
        score = self.calc_fid(fake_mean, fake_cov, real_mean, real_cov)
        print('=====> fid score: {}'.format(score))
        return score

    def calc_fid(self, fake_mean, fake_cov, real_mean, real_cov, eps=1e-6):
        cov_sqrt, _ = linalg.sqrtm(fake_cov @ real_cov, disp=False)
        print('calculating cov_sqrt was done')

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(fake_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((fake_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))
                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = fake_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(fake_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
        result = mean_norm + trace
        return result

    def load_pkl(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file, encoding='latin1')

    def save_pkl(self, obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--data_path", type=str, default='../dataset/endless_summer', help="学習に使用するデータセットのディレクトリを指定します")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=512, help="")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--resolution", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--pl_minibatch_shrink", type=int, default=1, help="pl_minibatch_shrink")
    parser.add_argument("--g_reg_interval", type=int, default=4, help="g_reg_interval")
    parser.add_argument("--d_reg_interval", type=int, default=16, help="d_reg_interval")
    parser.add_argument("--model_path", type=str, default='./model', help="model_path")
    parser.add_argument("--results", type=str, default='./results', help="results")
    parser.add_argument("--is_restore_model", type=strtobool, default=True, help="is_restore_model")
    parser.add_argument("--cache_path", type=str, default='./cache', help="cache")
    parser.add_argument("--tensorboard_path", type=str, default='./logs', help="tensorboard_path")
    opt = parser.parse_args()

    dataloader = get_dataloader(opt.data_path, opt.resolution, opt.batch_size)

    # dataset = BoxDataset(
    #     file_path=os.path.join(opt.data_path, '*.png'),
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(32),
    #             transforms.ToTensor(),
    #             TranformDynamicRange([0, 255], [-1, 1])
    #         ]
    #     ),
    # )
    #
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=8,
    #     shuffle=True,
    # )

    generator = Generator(opt)
    # if opt.is_restore_model:
    #     generator.restore()

    models = BaseLayer.restore_model(opt.model_path)
    if models is not None:
        generator, generator_predict, discriminator = models

    fid = FrechetInceptionDistance(generator, dataloader=dataloader, opt=opt)
    fid_score = fid.get_score()
    print(fid_score)
