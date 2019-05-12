# -*- coding: utf-8 -*-

import datetime
import math
import os
import time

import numpy as np
import scipy.io
import torch
from torch.autograd import Variable
import torch.nn.functional as f

import utils
from utils import AverageMeter
import tqdm


class Trainer(object):

    def __init__(self, cmd, cuda, model, optim=None,
                 train_loader=None, valid_loader=None, test_loader=None, log_file=None,
                 interval_validate=1, lr_scheduler=None,
                 start_step=0, total_steps=1e5, beta=0.05, start_epoch=0,
                 total_anneal_steps=200000, anneal_cap=0.2, do_normalize=True,
                 checkpoint_dir=None, result_dir=None, print_freq=1, result_save_freq=1, checkpoint_freq=1):

        self.cmd = cmd
        self.cuda = cuda
        self.model = model

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap

        self.n20_all = []
        self.n20_max_va, self.n100_max_va, self.r20_max_va, self.r50_max_va = 0, 0, 0, 0
        self.n20_max_te, self.n100_max_te, self.r20_max_te, self.r50_max_te = 0, 0, 0, 0


    def validate(self, cmd="valid"):
        assert cmd in ['valid', 'test']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        end = time.time()

        n20_list, n100_list, r20_list, r50_list = [], [], [], []

        loader_ = self.valid_loader if cmd == 'valid' else self.test_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, prof) in tqdm.tqdm(enumerate(loader_), total=len(loader_),
                                   desc='{} check epoch={}, len={}'.format('Valid' if cmd == 'valid' else 'Test',
                                                               self.epoch, len(loader_)), ncols=80, leave=False):
            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                if self.model.__class__.__name__ == 'MultiVAE':
                    logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
                else:
                    logits = self.model.forward(data_tr)
                pred_val = logits.cpu().detach().numpy()
                pred_val[data_tr.cpu().detach().numpy().nonzero()] = -np.inf

                n20_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te.numpy(), k=20))
                n100_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te.numpy(), k=100))
                r20_list.append(utils.Recall_at_k_batch(pred_val, data_te.numpy(), k=20))
                r50_list.append(utils.Recall_at_k_batch(pred_val, data_te.numpy(), k=50))

        n20_list = np.concatenate(n20_list, axis=0)
        n100_list = np.concatenate(n100_list, axis=0)
        r20_list = np.concatenate(r20_list, axis=0)
        r50_list = np.concatenate(r50_list, axis=0)
        
        
        if cmd == 'valid':
            self.n20_max_va = max(self.n20_max_va, n20_list.mean())
            self.n100_max_va = max(self.n100_max_va, n100_list.mean())
            self.r20_max_va = max(self.r20_max_va, r20_list.mean())
            self.r50_max_va = max(self.r50_max_va, r50_list.mean())
            max_metrics = "{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(cmd, self.epoch, self.step, self.n20_max_va, self.n100_max_va, self.r20_max_va, self.r50_max_va)

        else:
            self.n20_max_te = max(self.n20_max_te, n20_list.mean())
            self.n100_max_te = max(self.n100_max_te, n100_list.mean())
            self.r20_max_te = max(self.r20_max_te, r20_list.mean())
            self.r50_max_te = max(self.r50_max_te, r50_list.mean())
            max_metrics = "{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(cmd, self.epoch, self.step, self.n20_max_te, self.n100_max_te, self.r20_max_te, self.r50_max_te)

        metrics = []
        metrics.append(max_metrics)
        metrics.append("NDCG@20,{:.5f},{:.5f}".format(np.mean(n20_list), np.std(n20_list) / np.sqrt(len(n20_list))))
        metrics.append("NDCG@100,{:.5f},{:.5f}".format(np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
        metrics.append("Recall@20,{:.5f},{:.5f}".format(np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
        metrics.append("Recall@50,{:.5f},{:.5f}".format(np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
        print('\n' + ",".join(metrics))

        self.model.train()


    def train_epoch(self):
        cmd = "train"
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        for batch_idx, (data_tr, data_te, prof) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train check epoch={}, len={}'.format(self.epoch, len(self.train_loader)), ncols=80, leave=False):
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            if self.model.__class__.__name__ == 'MultiVAE':
                logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
            else:
                logits = self.model.forward(data_tr)

            log_softmax_var = f.log_softmax(logits, dim=1)
            neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))
            l2_reg = self.model.get_l2_reg()

            if self.model.__class__.__name__ == 'MultiVAE':
                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.step / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                loss = neg_ll + self.anneal * KL + l2_reg
                print("MultiVAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(), KL.cpu().detach().numpy(), l2_reg.cpu().detach().numpy() / 2, self.anneal, self.step, self.optim.param_groups[0]['lr'])
            else:
                loss = neg_ll + l2_reg
                print("MultiDAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(), l2_reg.cpu().detach().numpy() / 2, self.step)

            # backprop
            self.model.zero_grad()
            loss.backward()
            self.optim.step()

            if self.interval_validate > 0 and (self.step + 1) % self.interval_validate == 0:
                print("CALLING VALID", cmd, self.step, )
                self.validate()

    def train(self):
        max_epoch = 200
        for epoch in tqdm.trange(0, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.lr_scheduler.step()
            self.train_epoch()
            self.validate(cmd='valid')
            self.validate(cmd='test')

