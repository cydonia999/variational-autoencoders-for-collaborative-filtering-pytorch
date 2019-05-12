import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

import math
import utils
import numpy as np

__all__ = ['MultiVAE']

class Encoder(nn.Module):
    def __init__(self, options, dropout_p=0.5, q_dims=[20108, 600, 200]):
        super(Encoder, self).__init__()
        self.options = options
        self.q_dims = q_dims

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.linear_1 = nn.Linear(self.q_dims[0], self.q_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.q_dims[1], self.q_dims[2] * 2, bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        mu_q, logvar_q = torch.chunk(x, chunks=2, dim=1)
        return mu_q, logvar_q


class Decoder(nn.Module):
    def __init__(self, options, p_dims=[200, 600, 20108]):
        super(Decoder, self).__init__()
        self.options = options
        self.p_dims = p_dims

        self.linear_1 = nn.Linear(self.p_dims[0], self.p_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.p_dims[1], self.p_dims[2], bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        return x

class MultiVAE(nn.Module):
    def __init__(self, cuda2=True, weight_decay=0.0, dropout_p=0.5, q_dims=[20108, 600, 200], p_dims=[200, 600, 20108], n_conditioned=0):
        super(MultiVAE, self).__init__()
        self.cuda2 = cuda2
        self.weight_decay = weight_decay
        self.n_conditioned = n_conditioned
        self.q_dims = q_dims
        self.p_dims = p_dims
        self.q_dims[0] += self.n_conditioned
        self.p_dims[0] += self.n_conditioned

        self.encoder = Encoder(None, dropout_p=dropout_p, q_dims=self.q_dims)
        self.decoder = Decoder(None, p_dims=self.p_dims)

    def forward(self, x, c):
        x = f.normalize(x, p=2, dim=1)
        if self.n_conditioned > 0:
            x = torch.cat((x, c), dim=1)

        mu_q, logvar_q = self.encoder.forward(x)
        std_q = torch.exp(0.5 * logvar_q)
        KL = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), dim=1))
        epsilon = torch.randn_like(std_q, requires_grad=False)
        if True:
            if self.training:
                sampled_z = mu_q + epsilon * std_q
            else:
                sampled_z = mu_q
        else:
            sampled_z = mu_q + epsilon * std_q

        if self.n_conditioned > 0:
            sampled_z = torch.cat((sampled_z, c), dim=1)
        logits = self.decoder.forward(sampled_z)

        return logits, KL, mu_q, std_q, epsilon, sampled_z

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        if self.weight_decay > 0:
            for k, m in self.state_dict().items():
                if k.endswith('.weight'):
                    l2_reg = l2_reg + torch.norm(m, p=2) ** 2
        if self.cuda2:
            l2_reg = l2_reg.cuda()
        return self.weight_decay * l2_reg[0]
