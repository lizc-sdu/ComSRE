import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.utils import shuffle

import random
from tasks import *


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=False):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))

            # 仅在非最后一层添加 BN 和激活，保证 latent 层线性
            if i == self._dim - 1:
                continue

            if self._batchnorm:
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                    nn.Linear(decoder_dim[i], decoder_dim[i + 1]))

            # 最后一层（输出 x_hat）不添加激活/BN
            if i == self._dim - 1:
                continue
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


def device_as(t1, t2):
    return t1.to(t2.device)


class ComSRE(nn.Module):
    """ReCP module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        super(ComSRE, self).__init__()
        self._config = config
        self.city = config['city']
        self.embedding_size = config['Autoencoder']['arch1'][-1]

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        # View-specific autoencoders
        config['Autoencoder']['arch1'][0] = config[config['city']]['category_num']
        config['Autoencoder']['arch2'][0] = config[config['city']]['region_num'] + 1
        self.autoencoder_a = Autoencoder(config['Autoencoder']['arch1'],
                                         config['Autoencoder']['activations1'], config['Autoencoder']['batchnorm'])
        self.autoencoder_s = Autoencoder(config['Autoencoder']['arch2'],
                                         config['Autoencoder']['activations2'], config['Autoencoder']['batchnorm'])
        self.autoencoder_d = Autoencoder(config['Autoencoder']['arch2'],
                                         config['Autoencoder']['activations2'], config['Autoencoder']['batchnorm'])

        self.V = 3
        self.tau_I = 0.5
        self.latent_dim = config['Autoencoder']['arch1'][-1]
        self.con_dim = config['training']['con_dim']

        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.con_dim),
        )

        self.W_k = nn.Linear(self.latent_dim, self.latent_dim)
        self.W_v = nn.Linear(self.latent_dim, self.latent_dim)

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder_a.to(device)
        self.autoencoder_s.to(device)
        self.autoencoder_d.to(device)
        self.W_k.to(device)
        self.W_v.to(device)
        self.proj.to(device)

    def attention_fusion(self, view1, view2, view3):
        N = view1.shape[0]
        Z_list_encoded = [view1, view2, view3]
        Z_stack = torch.stack(Z_list_encoded, dim=1)

        q = torch.mean(Z_stack, dim=1)

        scores_list = []
        for v_idx in range(self.V):
            z_v = Z_list_encoded[v_idx]
            z_v_Wk = self.W_k(z_v)
            score_v = torch.sum(q * z_v_Wk, dim=1) / (self.latent_dim ** 0.5)
            scores_list.append(score_v)

        scores = torch.stack(scores_list, dim=1)

        A = F.softmax(scores, dim=1)
        projected_Z_for_c_list = [self.W_v(Z_v) for Z_v in Z_list_encoded]
        projected_Z_for_c_stack = torch.stack(projected_Z_for_c_list, dim=1)
        Z_con = torch.sum(A.unsqueeze(2) * projected_Z_for_c_stack, dim=1)
        return Z_con


    def contrastive_loss(self, view1, view2, view3, H_con, temperature=0.1, eps=1e-8):
        views = [view1, view2, view3]
        N = H_con.size(0)

        h_con_norm = F.normalize(H_con, p=2, dim=1)
        info_loss_total = 0.0

        for index, H_v in enumerate(views):
            h_v_norm = F.normalize(H_v, p=2, dim=1)
            similarity = torch.mm(h_con_norm, h_v_norm.t()) / temperature  # [N, N]
            exp_sim = torch.exp(similarity)

            pos = torch.diag(exp_sim)  # [N]
            denom = exp_sim.sum(dim=1)
            info_loss = -torch.mean(torch.log(pos / (denom + eps)))
            info_loss_total += info_loss

        info_loss_total = info_loss_total / len(views)

        return info_loss_total

    def uniqueness_loss(self, view1, view2, view3, H_con):
        U_list = [view1, view2, view3]
        loss_inter_unique_total = 0
        N = H_con.shape[0]

        for i in range(self.V):
            for j in range(i + 1, self.V):
                if i == 1 and j == 2:
                    continue
                input1 = U_list[i]
                input2 = U_list[j]
                batch_size = input1.size(0)
                input1 = input1.view(batch_size, -1)
                input2 = input2.view(batch_size, -1)

                # Zero mean
                input1_mean = torch.mean(input1, dim=0, keepdims=True)
                input2_mean = torch.mean(input2, dim=0, keepdims=True)
                input1 = input1 - input1_mean
                input2 = input2 - input2_mean

                input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
                input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

                input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
                input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

                diff_loss = torch.mean((input1_l2.t().mm(input2_l2) / N).pow(2))
                loss_inter_unique_total += diff_loss

        loss_inter_unique_total = loss_inter_unique_total / 2

        return loss_inter_unique_total

    def train(self, config, redata, xs,
              optimizer, device):
        for epoch in range(config['training']['epoch']):
            poi_emb = self.autoencoder_a.encoder(xs[0])
            s_emb = self.autoencoder_s.encoder(xs[1])
            d_emb = self.autoencoder_d.encoder(xs[2])

            # Attention fusion
            Z_con = self.attention_fusion(poi_emb, s_emb, d_emb)

            # Contrastive Loss
            H_poi = self.proj(poi_emb)
            H_s = self.proj(s_emb)
            H_d = self.proj(d_emb)
            H_con = self.proj(Z_con)

            cl_loss = self.contrastive_loss(H_poi, H_s, H_d, H_con)

            H_list_encoded = [H_poi, H_s, H_d]
            U_list = [(H_v - H_con) for H_v in H_list_encoded]
            loss_inter_unique = self.uniqueness_loss(
                U_list[0], U_list[1], U_list[2], H_con
            )

            # Reconstruction Loss
            recon1 = F.mse_loss(self.autoencoder_a.decoder(poi_emb), xs[0])
            recon2 = F.mse_loss(self.autoencoder_s.decoder(s_emb), xs[1])
            recon3 = F.mse_loss(self.autoencoder_d.decoder(d_emb), xs[2])
            recon_loss = (recon1 + recon2 + recon3) / 3

            loss = recon_loss + cl_loss * config[self.city]['lambda1'] + loss_inter_unique * \
                   config[self.city]['lambda2']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                print(
                    "Epoch : {:.0f}/{:.0f} ===>Loss = {:.4f}, ===>recon Loss = {:.4f}, ===>cl Loss = {:.4f},===>inter_unique Loss = {:.4f}".format(
                        (epoch + 1), config['training']['epoch'], loss, recon_loss, cl_loss,
                        loss_inter_unique))
                self.test(config, xs[0], xs[1], xs[2])

    def test(self, config, attribute_m, source_matrix, destina_matrix):
        with ((torch.no_grad())):
            self.autoencoder_a.eval(), self.autoencoder_s.eval(), self.autoencoder_d.eval(),
            self.W_k.eval(), self.W_v.eval(), self.proj.eval(),

            latent_a = self.autoencoder_a.encoder(attribute_m)
            latent_s = self.autoencoder_s.encoder(source_matrix)
            latent_d = self.autoencoder_d.encoder(destina_matrix)

            Z_con = self.attention_fusion(latent_a, latent_s, latent_d)  # Z_con: [N, d_z]

            H_poi = self.proj(latent_a)
            H_s = self.proj(latent_s)
            H_d = self.proj(latent_d)
            H_con = self.proj(Z_con)
            H_list_encoded = [H_poi, H_s, H_d]
            U_list_z = [(H_v - H_con) for H_v in H_list_encoded]

            region_emb = torch.cat([H_con, U_list_z[0], U_list_z[1], U_list_z[2]], dim=1).cpu().numpy()
            pre1, pre2, pre3 = self.predictions(region_emb)

            self.autoencoder_a.train(), self.autoencoder_s.train(), self.autoencoder_d.train(),
            self.W_k.train(), self.W_v.train(), self.proj.train(),

        return pre1, pre2, pre3, region_emb

    def predictions(self, region_emb):
        print("co2_sum Prediction")
        prediction1 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['co2_sum_path']),
                                   name='co2')
        print('==========================================>')
        print("gdp_sum Prediction")
        prediction2 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['gdp_sum_path']),
                                   name='gdp')
        print('==========================================>')
        print("population_sum Prediction")
        prediction3 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['population_sum_path']),
                                   name='popu')
        return prediction1, prediction2, prediction3
