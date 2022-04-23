#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 12/1/20
import sys
import os

from codes.utils.architectures import Architecture

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import pickle
from codes.utils.config import CONFIG_CANDIDATES

import traceback

import argparse
import logging
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from codes.utils.evaluate import evaluate
from codes.network.nasctr.network import DynamicNetwork, FixedNetwork
from codes.utils.train import train_search, train
from codes.utils.utils import NASCTR, count_parameters_in_mb, get_logger, PROJECT_PATH, check_directory, \
    set_seed, EarlyStop, PROJECT_PARENT_PATH, SEARCH_CHECKPOINT_PATH
from codes.utils.flops import get_model_params_flops


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', type=str, default='avazu', choices=['criteo', 'avazu', 'movielens', 'frappe', 'amazon'])
parser.add_argument('--dataset_path', type=str, default=f'{PROJECT_PATH}/codes/datasets/data/avazu.txt')
parser.add_argument('--search_train_ratio', type=float, default=0.75)
parser.add_argument('--search_valid_ratio', type=float, default=0.1)
parser.add_argument('--eval_train_ratio', type=float, default=0.8)
parser.add_argument('--eval_valid_ratio', type=float, default=0.1)
parser.add_argument('--search_batch_size', type=int, default=4096)
parser.add_argument('--evaluate_batch_size', type=int, default=4096)

# Adam
parser.add_argument('--adam_lr', type=float, default=0.001)
parser.add_argument('--adam_weight_decay', type=float, default=1e-6)
# SGD
parser.add_argument('--sgd_lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--sgd_lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--sgd_weight_decay', type=float, default=3e-4)
parser.add_argument('--sgd_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--sgd_decay_period', type=int, default=1, help='epochs between two learning rate decays')

# arch search
parser.add_argument('--use_pretrained_embeddings', type=bool, default=False)
parser.add_argument('--force_search', type=bool, default=True)
parser.add_argument('--arch_lr', type=float, default=0.001, choices=[0.001, 3e-4])
parser.add_argument('--arch_weight_decay', type=float, default=1e-6, choices=[1e-2, 1e-3, 1e-5, 1e-6])
parser.add_argument('--weights_reg', type=float, default=1e-5, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
parser.add_argument('--arch_reg', type=float, default=0.01, choices=[0, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 3])
parser.add_argument('--std_lambda', type=float, default=0)
parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--e_greedy', type=float, default=0)

parser.add_argument('--search_epochs', type=int, default=3)
parser.add_argument('--train_epochs', type=int, default=50)
parser.add_argument('--search_patience', type=int, default=5)
parser.add_argument('--eval_patience', type=int, default=5)

parser.add_argument('--block_in_dim', type=int, default=400)
parser.add_argument('--block_out_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--num_block', type=int, default=7)
parser.add_argument('--num_free_block', type=int, default=2)
parser.add_argument('--max_skip_connect', type=int, default=7)
parser.add_argument('--block_keys', type=list, default=None)

parser.add_argument('--mode', type=str, default='nasctr', help='choose how to search')

parser.add_argument('--gpu', type=int, default=3, help="gpu divice id")
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--seed', type=int, default=666, help="random seed")
parser.add_argument('--log_save_dir', type=str, default=PROJECT_PATH)
parser.add_argument('--show_log', type=bool, default=True)
parser.add_argument('--search_checkpoint_path', type=str,
                    default=os.path.join(PROJECT_PARENT_PATH, SEARCH_CHECKPOINT_PATH))

args = parser.parse_args()


def search_arch(dataset, train_queue, valid_queue, test_queue, args, LOGGER, check_existed=True):
    LOGGER.info(f"Start searching architecture...")
    search_start = time.time()
    if args.mode == 'nasctr':
        model = DynamicNetwork(dataset, args, logger=LOGGER)
        arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr, weight_decay=args.arch_weight_decay)
    else:
        raise Exception(f"No such mode {args.mode}")
    if args.use_gpu:
        model = model.cuda()

    LOGGER.info("Search total param size = %fMB", count_parameters_in_mb(model))

    search_checkpoint = os.path.join(args.search_checkpoint_path, model.get_tag() + '.pickle')
    try:
        if not args.force_search and check_existed and os.path.exists(search_checkpoint):
            with open(search_checkpoint, 'rb') as f:
                archs, use_time, mean_search_num_params, mean_test_auc, mean_test_loss = pickle.load(f)
            LOGGER.info(f"The model has been searched in {search_checkpoint}, use_time: {use_time}, mean_search_num_params: {mean_search_num_params}, mean_test_auc: {mean_test_auc}, mean_test_loss: {mean_test_loss}")
            return archs, use_time, mean_search_num_params, mean_test_auc, mean_test_loss
    except ValueError as e:
        pass

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.sgd_lr,
        momentum=args.sgd_momentum,
        weight_decay=args.sgd_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.search_epochs), eta_min=args.sgd_lr_min)
    best_auc, best_loss, best_arch = 0, float('inf'), None
    early_stopper = EarlyStop(k=args.search_patience)
    archs = []
    num_params_list = []
    test_auc_list = []
    test_loss_list = []
    for i in range(args.search_epochs):
        scheduler.step()
        LOGGER.info(f"search_epoch: {i}; lr: {scheduler.get_last_lr()}")
        train_loss, val_loss = train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
        model.binarize()
        test_loss, test_auc = evaluate(model, test_queue, args)
        arch, arch_p, num_params = model.get_arch()
        model.restore()

        archs.append((test_auc, test_loss, arch))
        num_params_list.append(num_params)
        test_auc_list.append(test_auc)
        test_loss_list.append(test_loss)

        LOGGER.info(
            f'search_epoch: {i}, train_loss: {train_loss:.5f},'
            f' val_loss: {val_loss:.5f},'
            f' test_loss: {test_loss:.5f}, test_auc: {test_auc:.5f},'
            f' time spent: {(time.time() - search_start):.5f},'
            f' arch: {arch}, num_params: {num_params}')
        if test_auc > best_auc or (test_auc == best_auc and test_loss < best_loss):
            best_auc = test_auc
            best_loss = test_loss
            best_arch = arch
        is_stop = early_stopper.add_metric(test_auc)
        # is_stop = early_stopper.add_metric(-test_loss)
        if is_stop:
            LOGGER.info(f"Not rise for {early_stopper.not_rise_times}, stop search")
            break
    mean_search_num_params = sum(num_params_list) / len(num_params_list)
    mean_test_auc = sum(test_auc_list) / len(test_auc_list)
    mean_test_loss = sum(test_loss_list) / len(test_loss_list)
    use_time = time.time() - search_start
    LOGGER.info(f"Finish searching archs, best auc: {best_auc}, best loss: {best_loss}, best_arch: {best_arch}, use_time: {use_time}, "
                f"mean_search_num_params: {mean_search_num_params}, mean_test_auc: {mean_test_auc}, mean_test_loss: {mean_test_loss}")
    archs.sort(key=lambda x: (-x[0], x[1]))

    with open(search_checkpoint, 'wb') as f:
        pickle.dump([archs, use_time, mean_search_num_params, mean_test_auc, mean_test_loss], f)

    return archs, use_time, mean_search_num_params, mean_test_auc, mean_test_loss


def evaluate_arch(dataset, train_queue, val_queue, test_queue, args, arch_info, LOGGER):
    auc, loss, arch = arch_info
    LOGGER.info(f"Start evaluate {arch}, search_auc: {auc}, search_loss: {loss}")
    eval_start = time.time()
    if args.mode == 'nasctr':
        model = FixedNetwork(dataset, args, arch, logger=LOGGER)
    else:
        raise Exception(f"No such mode {args.mode}")
    if args.use_gpu:
        model = model.cuda()

    params = count_parameters_in_mb(model)
    flops = get_model_params_flops(model, dataset, batch_size=1, use_gpu=args.use_gpu)
    LOGGER.info(f'Model={arch}, params={params}, flops={flops}')
    optimizer = torch.optim.Adam(model.parameters(), args.adam_lr, weight_decay=args.adam_weight_decay)
    try:
        val_auc, val_loss = train(train_queue, val_queue, model, optimizer, args, model_path=MODEL_PATH,
                                  patience=args.eval_patience, logger=LOGGER, show_log=args.show_log)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.pt')))
        test_loss, test_auc = evaluate(model, test_queue, args)
    except ValueError as e:
        test_loss, test_auc, flops, params, params_v1 = 100, 0, 0, 0, 0
        val_auc, val_loss = 0, 100
    LOGGER.info(f'Finish evaluate {arch}; '
                f'val_auc: {val_auc:.5f}, val_loss: {val_loss:.5f}; '
                f'test_auc: {test_auc:.5f}, test_loss: {test_loss:.5f}; '
                f'flops: {flops}, params: {params}; '
                f'time spent: {(time.time() - eval_start):.5f}')


def main(args, LOGGER):
    if args.use_gpu and not torch.cuda.is_available():
        LOGGER.error('no gpu device available')
        sys.exit(1)
    if args.use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True
        LOGGER.info('gpu device = %d' % args.gpu)
    LOGGER.info("args = %s", args)

    if args.dataset == 'criteo':
        from codes.datasets.criteo import CriteoDataset
        dataset = CriteoDataset(data_path=args.dataset_path, repreprocess=False,
                                use_pretrained_embeddings=args.use_pretrained_embeddings, logger=LOGGER)
    elif args.dataset == 'avazu':
        from codes.datasets.avazu import AvazuDataset
        dataset = AvazuDataset(data_path=args.dataset_path, repreprocess=False,
                               use_pretrained_embeddings=args.use_pretrained_embeddings, logger=LOGGER)
    elif args.dataset == 'movielens':
        from codes.datasets.movielens import MovielensDataset
        dataset = MovielensDataset(data_path=args.dataset_path, repreprocess=False,
                                   use_pretrained_embeddings=args.use_pretrained_embeddings, logger=None)
    elif args.dataset == 'frappe':
        from codes.datasets.frappe import FrappeDataset
        dataset = FrappeDataset(data_path=args.dataset_path, repreprocess=False,
                                use_pretrained_embeddings=args.use_pretrained_embeddings, logger=None)
    elif args.dataset == 'amazon':
        from codes.datasets.amazon import AmazonDataset
        dataset = AmazonDataset(data_path=args.dataset_path, repreprocess=False,
                                use_pretrained_embeddings=args.use_pretrained_embeddings, logger=None)
    else:
        raise Exception(f"No such dataset {args.dataset}!!!")

    train_queue, valid_queue, test_queue = dataset.get_search_dataloader(args.search_train_ratio,
                                                                         args.search_valid_ratio,
                                                                         args.search_batch_size)
    archs, use_time, mean_search_num_params, mean_test_auc, mean_test_loss =\
        search_arch(dataset, train_queue, valid_queue, test_queue, args, LOGGER)
    # archs = [(0, 0, Architecture(blocks=[('ElementWise-min', [0, 1]), ('ElementWise-max', [1, 2]), ('MLP-64', [3, 2]), ('MLP-1024', [4, 1]), ('FM', [3, 1]), ('FM', [4, 2]), ('MLP-256', [7, 2])], concat=range(2, 9)))]

    args.use_pretrained_embeddings = False
    test_queue = dataset.get_test_dataloader(args.evaluate_batch_size)
    train_queue, val_queue = dataset.get_eval_dataloader(args.eval_train_ratio, args.eval_valid_ratio,
                                                         args.evaluate_batch_size)
    evaluate_arch(dataset, train_queue, val_queue, test_queue, args, archs[0], LOGGER)
    LOGGER.info(f"Search info use_time={use_time}, mean_search_num_params={mean_search_num_params}, mean_test_auc={mean_test_auc}, {mean_test_loss}=mean_test_loss")


if __name__ == '__main__':
    set_seed(args.seed)
    search_checkpoint_path = os.path.join(PROJECT_PARENT_PATH, SEARCH_CHECKPOINT_PATH, args.dataset,
                                          str(args.seed), str(args.use_pretrained_embeddings))
    args.search_checkpoint_path = os.path.join(search_checkpoint_path, args.mode)
    check_directory(args.search_checkpoint_path, force_removed=False)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    MODEL_PATH = f'model_checkpoint/{args.dataset}/{timestamp}'
    check_directory(MODEL_PATH, force_removed=True)
    log_root_path = os.path.join(PROJECT_PATH, 'logs')
    log_parent_dir = os.path.join(log_root_path,
                                  args.dataset,
                                  str(args.seed),
                                  f'{args.num_block}_{args.num_free_block}_{args.max_skip_connect}_{args.embedding_dim}_{args.block_in_dim}_{args.arch_reg}_{str(args.use_pretrained_embeddings)}_{args.weights_reg}_{args.std_lambda}',
                                  args.mode)
    check_directory(log_parent_dir, force_removed=True)
    log_save_dir = os.path.join(log_parent_dir,
                                f'log_search_{args.dataset}_{args.mode}_{args.seed}.txt')
    LOGGER = get_logger(NASCTR + str(timestamp), log_save_dir, level=logging.DEBUG)
    try:
        main(args, LOGGER)
    except Exception as e:
        LOGGER.error(traceback.format_exc())

