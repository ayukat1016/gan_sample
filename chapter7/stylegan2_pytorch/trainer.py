import os
import math
import numpy as np
import torch
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
from base_layer import BaseLayer
from generator_loss import GeneratorLoss
from generator_loss_pathreg import GeneratorLossPathReg
from discriminator_loss import DiscriminatorLoss
from discriminator_loss_r1 import DiscriminatorLossR1
from datasource import get_dataloader
from frechet_Inception_distance import FrechetInceptionDistance
from tensorboard_logger import TensorboardLogger
from utility import adjust_dynamic_range, save_image_grid
from transform import TranformDynamicRange
import torch.nn.functional as F

class Trainer:
    def __init__(self, opt):
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.opt = opt
        self.generator = Generator(opt)
        self.generator_predict = Generator(opt)
        self.discriminator = Discriminator(opt)

        self.decay = 0.5**(opt.batch_size / (10 * 1000)) * opt.adjust_decay_param
        first_decay = 0
        if opt.is_restore_model:
            models = BaseLayer.restore_model(opt.model_path)
            if models is not None:
                self.generator, self.generator_predict, self.discriminator = models
                first_decay = self.decay

        BaseLayer.print_model_parameters(self.generator, 'generator')
        BaseLayer.print_model_parameters(self.discriminator, 'discriminator')

        self.generator.train()
        self.generator_predict.eval()

        Generator.apply_decay_parameters(self.generator, self.generator_predict, decay=first_decay)
        self.discriminator.train()

        self.generator_loss = GeneratorLoss()
        self.generator_loss_path_reg = GeneratorLossPathReg(opt=opt)

        self.discriminator_loss = DiscriminatorLoss()
        self.discriminator_loss_r1 = DiscriminatorLossR1(reg_interval=opt.d_reg_interval)

        self.dataloader = get_dataloader(opt.data_path, opt.resolution, opt.batch_size)
        self.fid = FrechetInceptionDistance(self.generator_predict, self.dataloader, opt)

        learning_rate, beta1, beta2 = self.get_adam_params_adjust_interval(opt.g_reg_interval, opt)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        learning_rate, beta1, beta2 = self.get_adam_params_adjust_interval(opt.d_reg_interval, opt)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        if not os.path.isdir(self.opt.cache_path):
            os.makedirs(self.opt.cache_path, exist_ok=True)

    def train_generator(self, current_loop_num):
        BaseLayer.set_model_parameter_requires_grad_all(self.generator, True)
        BaseLayer.set_model_parameter_requires_grad_all(self.discriminator, False)

        # train generator
        # TensorboardLogger.print_parameter(generator)
        for index in range(0, self.opt.generator_train_num):
            train_z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
            fake_imgs, fake_dlatents_out = self.generator(train_z)
            fake_validity = self.discriminator(fake_imgs)

            prob_fake = F.sigmoid(fake_validity).mean()
            TensorboardLogger.write_scalar('prob_fake/generator', prob_fake)
            # print('{} prob_fake(generator): {}'.format(index, prob_fake))

            g_loss = self.generator_loss(fake_validity)
            self.optimizer_g.zero_grad()
            g_loss.backward()
            self.optimizer_g.step()

        run_g_reg = current_loop_num % self.opt.g_reg_interval == 0
        if run_g_reg:
            # generatorの正則化処理
            g_reg_maxcount = 4 if 4 < self.opt.generator_train_num else self.opt.generator_train_num
            for _ in range(0, g_reg_maxcount):
                z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
                pl_fake_imgs, pl_fake_dlatents_out = self.generator(z)
                g_reg, pl_lenght = self.generator_loss_path_reg(pl_fake_imgs, pl_fake_dlatents_out)
                self.optimizer_g.zero_grad()
                g_reg.backward()
                self.optimizer_g.step()

            TensorboardLogger.write_scalar('loss/g_reg', g_reg)
            TensorboardLogger.write_scalar('loss/path_length', pl_lenght)
            TensorboardLogger.write_scalar('loss/pl_mean_var', self.generator_loss_path_reg.pl_mean_var.mean())

        # 推論用のgeneratorに指数移動平均を行った重みを適用する
        Generator.apply_decay_parameters(self.generator, self.generator_predict, decay=self.decay)
        fake_imgs_predict, fake_dlatents_out_predict = self.generator_predict(train_z)
        fake_predict_validity = self.discriminator(fake_imgs_predict)
        prob_fake_predict = F.sigmoid(fake_predict_validity).mean()
        TensorboardLogger.write_scalar('prob_fake_predict/generator', prob_fake_predict)
        # print('prob_fake_predict(generator): {}'.format(prob_fake_predict))

        Generator.apply_decay_parameters(self.generator_predict, self.generator, decay=self.opt.reverse_decay)

        if current_loop_num % self.opt.save_metrics_interval == 0:
            TensorboardLogger.write_scalar('score/g_score', fake_validity.mean())
            TensorboardLogger.write_scalar('loss/g_loss', g_loss)
            TensorboardLogger.write_histogram('generator/fake_imgs', fake_imgs)
            TensorboardLogger.write_histogram('generator/fake_dlatents_out', fake_dlatents_out)
            TensorboardLogger.write_histogram('generator/fake_imgs_predict', fake_imgs_predict)
            TensorboardLogger.write_histogram('generator/fake_dlatents_out_predict', fake_dlatents_out_predict)

        if current_loop_num % self.opt.save_images_tensorboard_interval == 0:
            # for index in range(fake_imgs.shape[0]):
            #     img = adjust_dynamic_range(fake_imgs[index].to('cpu').detach().numpy(), drange_in=[-1, 1], drange_out=[0, 255])
            #     TensorboardLogger.write_image('images/fake/{}'.format(index), img)

            for index in range(fake_imgs_predict.shape[0]):
                img = adjust_dynamic_range(fake_imgs_predict[index].to('cpu').detach().numpy(), drange_in=[-1, 1], drange_out=[0, 255])
                TensorboardLogger.write_image('images/fake_predict/{}'.format(index), img)

        if current_loop_num % self.opt.save_images_interval == 0:
            # 生成した画像を保存する
            if not os.path.isdir(self.opt.results):
                os.makedirs(self.opt.results, exist_ok=True)
            # fake_imgs_val, fake_dlatents_out_val = generator(val_z)
            # save_image_grid(
            #     # fake_imgs_val.to('cpu').detach().numpy(),
            #     fake_imgs.to('cpu').detach().numpy(),
            #     os.path.join(self.opt.results, '{}_fake.png'.format(TensorboardLogger.global_step)),
            #     batch_size=self.opt.batch_size,
            #     drange=[-1, 1])

            # fake_imgs_predict_val, fake_dlatents_out_predict_val = generator_predict(val_z)
            save_image_grid(
                fake_imgs_predict.to('cpu').detach().numpy(),
                os.path.join(self.opt.results, '{}_fake_predict.png'.format(TensorboardLogger.global_step)),
                batch_size=self.opt.batch_size,
                drange=[-1, 1])

        return g_loss

    def train_discriminator(self, current_loop_num):
        BaseLayer.set_model_parameter_requires_grad_all(self.generator, False)
        BaseLayer.set_model_parameter_requires_grad_all(self.discriminator, True)

        # train discriminator
        for index in range(0, self.opt.discriminator_train_num):
            data_iterator = self.dataloader.__iter__()
            imgs = data_iterator.next()
            # imgs = TranformDynamicRange.fade_lod(x=imgs, lod=0.0)
            # imgs = TranformDynamicRange.upscale_lod(x=imgs, lod=0.0)
            real_imgs = Variable(imgs.type(self.Tensor), requires_grad=False)

            z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
            fake_imgs, fake_dlatents_out = self.generator(z)

            real_validity = self.discriminator(real_imgs)
            prob_real = F.sigmoid(real_validity).mean()
            TensorboardLogger.write_scalar('prob_real/discriminator', prob_real)
            # print('{} prob_real(discriminator): {}'.format(index, prob_real))

            fake_validity = self.discriminator(fake_imgs)
            prob_fake = F.sigmoid(fake_validity).mean()
            TensorboardLogger.write_scalar('prob_fake/discriminator', prob_fake)
            # print('{} prob_fake(discriminator): {}'.format(index, prob_fake))

            d_loss = self.discriminator_loss(fake_validity, real_validity)
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        run_d_reg = current_loop_num % self.opt.d_reg_interval == 0
        if run_d_reg:
            d_reg_maxcount = 4 if 4 < self.opt.discriminator_train_num else self.opt.discriminator_train_num
            for index in range(0, d_reg_maxcount):
                # discriminatorの正則化処理
                # z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
                # fake_imgs, fake_dlatents_out = self.generator(z)
                # fake_validity = self.discriminator(fake_imgs)

                real_imgs.requires_grad = True
                real_validity = self.discriminator(real_imgs)

                d_reg = self.discriminator_loss_r1(real_validity, real_imgs)
                self.optimizer_d.zero_grad()
                d_reg.backward()
                self.optimizer_d.step()
            TensorboardLogger.writer.add_scalar('{}/reg/d_reg'.format(TensorboardLogger.now), d_reg, TensorboardLogger.global_step)

        if current_loop_num % self.opt.save_metrics_interval == 0:
            TensorboardLogger.write_scalar('score/d_score', real_validity.mean())
            TensorboardLogger.write_scalar('loss/d_loss', d_loss)
            TensorboardLogger.write_histogram('real_imgs', real_imgs)

        return d_loss

    def save_model(self):
        BaseLayer.save_model(self.opt.model_path, (self.generator, self.generator_predict, self.discriminator))

    def calculate_fid_score(self):
        fid_score = self.fid.get_score()
        TensorboardLogger.write_scalar('score/fid', fid_score)

    def get_adam_params_adjust_interval(self, reg_interval, opt):
        minibatch_ratio = reg_interval / (reg_interval + 1)
        l_rate = opt.learning_rate * minibatch_ratio
        b1 = opt.beta1**minibatch_ratio
        b2 = opt.beta2**minibatch_ratio
        return l_rate, b1, b2
