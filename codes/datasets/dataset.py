#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import shutil

import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader

from codes.utils.utils import get_logger, NASCTR

PRE_DENSE_FEATES_FILE = 'pre_dense_feats.npy'
PRE_SPARSE_FEATES_FILE = 'pre_sparse_feats.npy'
LABELS_FEATES_FILE = 'labels.npy'

DENSE_TRAIN = 'splits/pre_dense_feats_train.npy'
DENSE_TEST = 'splits/pre_dense_feats_test.npy'
SPARSE_TRAIN = 'splits/pre_sparse_feats_train.npy'
SPARSE_TEST = 'splits/pre_sparse_feats_test.npy'
LABEL_TRAIN = 'splits/labels_train.npy'
LABEL_TEST = 'splits/labels_test.npy'

FIELD_DIMS_FILE = 'field_dims.npy'
PRETRAINED_EMBEDDINGS_DIR = 'pretrained_embeddings'
BEST_PRETRAINED_EMBEDDINGS_FILE = 'pretrained_embeddings_best.pt'
FINAL_PRETRAINED_EMBEDDINGS_FILE = 'pretrained_embeddings_final.pt'


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TestDataset(data.Dataset):
    def __init__(self, dense, sparse, label):
        self._dense = dense
        self._sparse = sparse
        self._label = label
        self._num_data = len(self._label)

    def __getitem__(self, index):
        return self._dense[index], self._sparse[index], self._label[index]

    def __len__(self):
        return self._num_data

    @property
    def num_data(self):
        return self._num_data


class Dataset(data.Dataset):
    def __init__(self, data_path, num_feats, num_dense_feats, num_sparse_feats, repreprocess=False, use_pretrained_embeddings=True, with_head=False, logger=None):
        self._data_path = data_path
        self._save_path = os.path.dirname(data_path)
        self._repreprocess = repreprocess
        self._with_head = with_head
        self._use_pretrained_embeddings = use_pretrained_embeddings
        self._pretrained_embeddings = None

        self._num_feats = num_feats
        self._num_dense_feats = num_dense_feats
        self._num_sparse_feats = num_sparse_feats
        self._num_data = None
        self._num_train_data = None
        self._num_test_data = None
        self._test_dataset = None
        # including one-hot features for sparse features
        self._field_dims = np.zeros(self._num_feats, dtype=np.int64)
        self._num_total_fields_dims = None

        self._dense_feats, self._sparse_feats, self._labels = None, None, None

        if logger is None:
            self._logger = get_logger(NASCTR)
        else:
            self._logger = logger

    def __getitem__(self, index):
        return self._dense_feats[index], self._sparse_feats[index], self._labels[index]

    def __len__(self):
        return self._num_train_data

    def get_search_dataloader(self, train_ratio, valid_ratio, batch_size, num_workers=4, random_val=True):
        train_num = int(self._num_data * train_ratio)
        valid_num = int(self._num_data * valid_ratio)
        test_num = self._num_train_data - train_num - valid_num

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            self._logger.info(f"Cpu count {num_workers}")

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(self, (train_num, valid_num, test_num))

        train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(train_num))))
        test_dataloader = DataLoaderX(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        sampler = None
        if random_val:
            num_samples = (train_num // batch_size + 1) * batch_size
            sampler = data.sampler.RandomSampler(valid_dataset, replacement=True, num_samples=num_samples)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=True, sampler=sampler)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_eval_dataloader(self, train_ratio, valid_ratio, batch_size, num_workers=4, shuffle=True):
        train_num = int(self._num_data * train_ratio)
        valid_num = int(self._num_train_data - train_num)

        indices = list(range(self._num_train_data))
        if shuffle:
            np.random.shuffle(indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:train_num])
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[train_num:])

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            self._logger.info(f"Cpu count {num_workers}")

        train_dataloader = DataLoaderX(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
        val_dataloader = DataLoaderX(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=val_sampler)
        return train_dataloader, val_dataloader

    def get_test_dataloader(self, batch_size, num_workers=4):
        test_dataloader = DataLoaderX(self._test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        return test_dataloader

    def sample_a_batch(self, batch_size):
        test_dataloader = DataLoaderX(self._test_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
        for dense_train, sparse_train, labels_train in test_dataloader:
            return dense_train, sparse_train, labels_train

    def _check_preprocess_file(self):
        return os.path.exists(os.path.join(self._save_path, PRE_DENSE_FEATES_FILE)) and \
               os.path.exists(os.path.join(self._save_path, PRE_SPARSE_FEATES_FILE)) and \
               os.path.exists(os.path.join(self._save_path, LABELS_FEATES_FILE)) and \
               os.path.exists(os.path.join(self._save_path, FIELD_DIMS_FILE)) and \
               os.path.exists(os.path.join(self._save_path, DENSE_TRAIN)) and \
               os.path.exists(os.path.join(self._save_path, DENSE_TEST)) and \
               os.path.exists(os.path.join(self._save_path, SPARSE_TRAIN)) and \
               os.path.exists(os.path.join(self._save_path, SPARSE_TEST)) and \
               os.path.exists(os.path.join(self._save_path, LABEL_TRAIN)) and \
               os.path.exists(os.path.join(self._save_path, LABEL_TEST)) and \
               os.path.exists(os.path.join(self._save_path, FIELD_DIMS_FILE))

    def _split_dataset(self):
        dense_train, dense_test, sparse_train, sparse_test, labels_train, labels_test = \
            train_test_split(self._dense_feats, self._sparse_feats, self._labels, test_size=0.1)

        path = os.path.join(self._save_path, 'splits')
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass
        os.makedirs(path)
        np.save(os.path.join(self._save_path, DENSE_TRAIN), dense_train)
        np.save(os.path.join(self._save_path, DENSE_TEST), dense_test)
        np.save(os.path.join(self._save_path, SPARSE_TRAIN), sparse_train)
        np.save(os.path.join(self._save_path, SPARSE_TEST), sparse_test)
        np.save(os.path.join(self._save_path, LABEL_TRAIN), labels_train)
        np.save(os.path.join(self._save_path, LABEL_TEST), labels_test)

    def _read(self):
        if self._check_preprocess_file():
            self._logger.info("Use preprocessed dataset")
            self._dense_feats = np.load(os.path.join(self._save_path, DENSE_TRAIN))
            self._sparse_feats = np.load(os.path.join(self._save_path, SPARSE_TRAIN))
            self._labels = np.load(os.path.join(self._save_path, LABEL_TRAIN))
            self._field_dims = np.load(os.path.join(self._save_path, FIELD_DIMS_FILE))
            dense_test = np.load(os.path.join(self._save_path, DENSE_TEST))
            sparse_test = np.load(os.path.join(self._save_path, SPARSE_TEST))
            labels_test = np.load(os.path.join(self._save_path, LABEL_TEST))
            self._test_dataset = TestDataset(dense_test, sparse_test, labels_test)
        if os.path.exists(os.path.join(self._save_path, PRETRAINED_EMBEDDINGS_DIR, BEST_PRETRAINED_EMBEDDINGS_FILE)) and self._use_pretrained_embeddings:
            self._pretrained_embeddings = torch.load(os.path.join(self._save_path, PRETRAINED_EMBEDDINGS_DIR, BEST_PRETRAINED_EMBEDDINGS_FILE))
        else:
            self._use_pretrained_embeddings = False

        self._num_train_data = len(self._labels)
        self._num_test_data = self._test_dataset.num_data
        self._num_data = len(self._labels)
        self._num_total_fields_dims = sum(self._field_dims)

        self._logger.info(f"Total data {self._num_data}; "
                          f"Total Train data {self._num_train_data}; "
                          f"Total Test data {self._num_test_data}; "
                          f"Total feats {self._num_feats}; "
                          f"Dense feats {self._num_dense_feats}; "
                          f"Sparse feats {self._num_sparse_feats}; "
                          f"Field dims {len(self._field_dims)} {self._field_dims};"
                          f"Total feats with one-hot {self._num_total_fields_dims}")

    def _transform_feat(self, *args, **kwargs):
        raise NotImplementedError

    def _get_feat_dict(self):
        raise NotImplementedError

    @property
    def num_feats(self):
        return self._num_feats

    @property
    def num_dense_feats(self):
        return self._num_dense_feats

    @property
    def num_sparse_feats(self):
        return self._num_sparse_feats

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_total_fields_dims(self):
        return self._num_total_fields_dims

    @property
    def dense_feats(self):
        return self._dense_feats

    @property
    def sparse_feats(self):
        return self._sparse_feats

    @property
    def labels(self):
        return self._labels

    @property
    def field_dims(self):
        return self._field_dims

    @property
    def save_path(self):
        return self._save_path

    @property
    def pretrained_embeddings(self):
        return self._pretrained_embeddings
