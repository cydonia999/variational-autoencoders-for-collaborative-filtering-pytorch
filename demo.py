#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import scipy.io
import numpy as np

import torch
import dataset
from vae import MultiVAE
from dae import MultiDAE
from trainer import Trainer
import utils
import tqdm
import pandas as pd


configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=200000, # "lr_policy: step" e-6
        interval_validate=1000,
    ),
}

def main():
    parser = argparse.ArgumentParser("Variational autoencoders for collaborative filtering")
    parser.add_argument('cmd', type=str,  choices=['train'], help='train')
    parser.add_argument('--arch_type', type=str, default='MultiVAE', help='architecture', choices=['MultiVAE', 'MultiDAE'])
    parser.add_argument('--dataset_name', type=str, default='ml-20m', help='camera model type', choices=['ml-20m', 'lastfm-360k'])
    parser.add_argument('--processed_dir', type=str, default='O:/dataset/vae_cf/data/ml-20m/pro_sg', help='dataset directory')
    parser.add_argument('--n_items', type=int, default=1, help='n items')
    parser.add_argument('--conditioned_on', type=str, default=None, help='conditioned on user profile (g: gender, a: age, c: country) for Last.fm')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='checkpoints directory')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='checkpoint save frequency')
    parser.add_argument('--valid_freq', type=int, default=1, help='validation frequency in training')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--start_step', dest='start_step', type=int, default=0, help='start step')
    parser.add_argument('--total_steps', dest='total_steps', type=int, default=int(3e5), help='Total number of steps')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='batch size in validation')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size in test')
    parser.add_argument('--print_freq', type=int, default=1, help='log print frequency')
    parser.add_argument('--upper_train', type=int, default=-1, help='max of train images(for debug)')
    parser.add_argument('--upper_valid', type=int, default=-1, help='max of valid images(for debug)')
    parser.add_argument('--upper_test', type=int, default=-1, help='max of test images(for debug)')
    parser.add_argument('--total_anneal_steps', type=int, default=0, help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
    parser.add_argument('--dropout_p', dest='dropout_p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    args = parser.parse_args()

    if args.cmd == 'train':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        cfg = configurations[args.config]

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(98765)
    if cuda:
        torch.cuda.manual_seed(98765)

    # # 1. data loader
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    root = args.processed_dir

    DS = dataset.MovieLensDataset if args.dataset_name == 'ml-20m' else dataset.LastfmDataset
    if args.cmd == 'train':
        dt = DS(root, 'data_csr.pkl', split='train', upper=args.upper_train, conditioned_on=args.conditioned_on)
        train_loader = torch.utils.data.DataLoader(dt, batch_size=args.train_batch_size, shuffle=True, **kwargs)

        dt = DS(root, 'data_csr.pkl', split='valid', upper=args.upper_valid, conditioned_on=args.conditioned_on)
        valid_loader = torch.utils.data.DataLoader(dt, batch_size=args.valid_batch_size, shuffle=False, **kwargs)

        dt = DS(root, 'data_csr.pkl', split='test', upper=args.upper_test, conditioned_on=args.conditioned_on)
        test_loader = torch.utils.data.DataLoader(dt, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # 2. model
    n_conditioned = 0
    if args.conditioned_on: # used for conditional VAE
        if 'g' in args.conditioned_on:
            n_conditioned += 3
        if 'a' in args.conditioned_on:
            n_conditioned += 10
        if 'c' in args.conditioned_on:
            n_conditioned += 17

    if 'MultiVAE' in args.arch_type:
        model = MultiVAE(dropout_p=args.dropout_p, weight_decay=0.0, cuda2=cuda,
                         q_dims=[args.n_items, 600, 200], p_dims=[200, 600, args.n_items], n_conditioned=n_conditioned)
    if 'MultiDAE' in args.arch_type:
        model = MultiDAE(dropout_p=args.dropout_p, weight_decay=0.01 / args.train_batch_size, cuda2=cuda)
    print(model)

    start_epoch = 0
    start_step = 0

    if cuda:
        model = model.cuda()

    # 3. optimizer
    if args.cmd == 'train':
        optim = torch.optim.Adam(
            [
                {'params': list(utils.get_parameters(model, bias=False)), 'weight_decay': 0.0},
                {'params': list(utils.get_parameters(model, bias=True)), 'weight_decay': 0.0},
            ],
            lr=cfg['lr'],
        )

        # lr_policy: step
        last_epoch = -1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 150], gamma=cfg['gamma'], last_epoch=last_epoch)

    if args.cmd == 'train':
        trainer = Trainer(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            optim=optim,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            start_step=start_step,
            total_steps=args.total_steps,
            interval_validate=args.valid_freq,
            checkpoint_dir=args.checkpoint_dir,
            print_freq=args.print_freq,
            checkpoint_freq=args.checkpoint_freq,
            total_anneal_steps=args.total_anneal_steps,
            anneal_cap=args.anneal_cap,
        )
        trainer.train()


if __name__ == '__main__':
    main()
