import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

import math
import utils
import numpy as np

__all__ = ['MultiDAE']

class Encoder(nn.Module):
    def __init__(self, options, dropout_p=0.5, q_dims=[20108, 200]):
        super(Encoder, self).__init__()
        self.options = options
        self.q_dims = q_dims

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.linear_1 = nn.Linear(self.q_dims[0], self.q_dims[1], bias=True)
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
        return x


class Decoder(nn.Module):
    def __init__(self, options, p_dims=[200, 20108]):
        super(Decoder, self).__init__()
        self.options = options
        self.p_dims = p_dims

        self.linear_1 = nn.Linear(self.p_dims[0], self.p_dims[1], bias=True)
        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.linear_1(x)
        return x

class MultiDAE(nn.Module):
    def __init__(self, cuda2=True, weight_decay=0.0, dropout_p=0.5, q_dims=[20108, 200], p_dims=[200, 20108]):
        super(MultiDAE, self).__init__()
        self.cuda2 = cuda2
        self.weight_decay = weight_decay

        self.encoder = Encoder(None, dropout_p=dropout_p, q_dims=q_dims)
        self.decoder = Decoder(None, p_dims=p_dims)

    def forward(self, x):
        x = f.normalize(x, p=2, dim=1)
        x = self.encoder.forward(x)
        logits = self.decoder.forward(x)
        return logits

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        if self.weight_decay > 0:
            for k, m in self.state_dict().items():
                if k.endswith('.weight'):
                    l2_reg = l2_reg + torch.norm(m, p=2) ** 2
            l2_reg = self.weight_decay * l2_reg
        if self.cuda2:
            l2_reg = l2_reg.cuda()
        return l2_reg[0]

