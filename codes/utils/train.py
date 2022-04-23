#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from codes.utils.evaluate import evaluate
from codes.utils.utils import EarlyStop


def train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args, log_interval=100):
    model.train()
    train_losses = []
    valid_losses = []

    total_train_loss = 0
    total_valid_loss = 0
    tt = tqdm(train_queue, smoothing=0, mininterval=1.0)
    valid_queue_iter = iter(valid_queue)
    for i, (dense_train, sparse_train, labels_train) in enumerate(tt):
        dense_valid, sparse_valid, labels_valid = next(valid_queue_iter)
        if args.use_gpu:
            dense_train = dense_train.cuda(non_blocking=True)
            sparse_train = sparse_train.cuda(non_blocking=True)
            labels_train = labels_train.cuda(non_blocking=True)
            dense_valid = dense_valid.cuda(non_blocking=True)
            sparse_valid = sparse_valid.cuda(non_blocking=True)
            labels_valid = labels_valid.cuda(non_blocking=True)
        if args.mode == 'no-auto':
            # TODO 这里的 lr 值
            loss_valid = model.step(dense_train, sparse_train, labels_train,
                                    dense_train, sparse_train, labels_train,
                                    arch_optimizer)
        else:
            loss_valid = model.step(dense_train, sparse_train, labels_train,
                                    dense_valid, sparse_valid, labels_valid,
                                    arch_optimizer)
        valid_loss = loss_valid.cpu().detach().item()
        optimizer.zero_grad()
        arch_optimizer.zero_grad()

        model.binarize(args.e_greedy)
        predicts, regs = model((dense_train, sparse_train))
        train_loss = model.compute_loss(predicts, labels_train, regs, use_arch_loss=False)
        train_loss.backward()
        model.restore()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.clip()
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        train_loss = train_loss.cpu().detach().item()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        total_train_loss += train_loss
        total_valid_loss += valid_loss
        if (i + 1) % log_interval == 0:
            tt.set_postfix(val_loss=total_valid_loss / log_interval, train_loss=total_train_loss / log_interval)
            total_train_loss = 0
            total_valid_loss = 0

    return np.mean(train_losses), np.mean(valid_losses)


def train(train_queue, val_queue, model, optimizer, args, model_path=None, patience=10,
          log_interval=100, logger=None, show_log=True):
    early_stopper = EarlyStop(k=patience)
    losses = []
    start = time.time()
    best_auc, best_loss = 0, float('inf')
    model.train()
    for train_epoch in range(args.train_epochs):
        cur_epoch_losses = []
        total_train_loss = 0
        tt = tqdm(train_queue, smoothing=0, mininterval=1.0)
        for i, (dense_train, sparse_train, labels_train) in enumerate(tt):
            if args.use_gpu:
                dense_train = dense_train.cuda(non_blocking=True)
                sparse_train = sparse_train.cuda(non_blocking=True)
                labels_train = labels_train.cuda(non_blocking=True)
            predicts, regs = model((dense_train, sparse_train))
            if args.mode not in ['darts', 'rl']:
                loss = model.compute_loss(predicts, labels_train, regs, use_reg=True, use_arch_loss=False)
            else:
                loss = model.compute_loss(predicts, labels_train, regs, use_reg=True)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.cpu().detach().item()
            cur_epoch_losses.append(loss)

            total_train_loss += loss
            if (i + 1) % log_interval == 0:
                tt.set_postfix(train_epoch=train_epoch, train_loss=total_train_loss / log_interval)
                total_train_loss = 0
        losses.append(np.mean(cur_epoch_losses))

        val_loss, val_auc = evaluate(model, val_queue, args)
        is_stop = early_stopper.add_metric(val_auc)
        if val_auc > best_auc or (val_auc == best_auc and val_loss < best_loss):
            if model_path is not None:
                torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
            best_auc = val_auc
            best_loss = val_loss
        if show_log:
            logger.info(f'train_epoch: {train_epoch}, train_loss: {losses[-1]:.5f}, '
                        f'val_auc: {val_auc:.5f}, val_loss: {val_loss:.5f}, '
                        f'time: {(time.time() - start):.5f} ')
        if is_stop:
            logger.info(f"Not rise for {early_stopper.not_rise_times}, stop train")
            break

    return best_auc, best_loss
