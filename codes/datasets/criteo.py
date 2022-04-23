#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
sys.path.append('../../')

import math
from collections import defaultdict
from functools import lru_cache

from tqdm import tqdm

from codes.utils.utils import get_logger, NASCTR, PROJECT_PATH

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from codes.datasets.dataset import Dataset, PRE_DENSE_FEATES_FILE, PRE_SPARSE_FEATES_FILE, LABELS_FEATES_FILE, \
    FIELD_DIMS_FILE


@lru_cache(maxsize=None)
def scale(x):
    try:
        x = int(x)
    except Exception as e:
        # TODO 空值赋成0是否科学
        return 0
    if x > 2:
        x = int(math.log(x) ** 2)
    return x


class CriteoDataset(Dataset):
    def __init__(self, data_path, repreprocess=False, use_pretrained_embeddings=True, logger=None):
        """
        Extra preprocess steps:
        1. remove the infrequent features
        2. discretize numerical values

        Dataset info:
            0.label
            1.I1
            2.I2
            3.I3
            4.I4
            5.I5
            6.I6
            7.I7
            8.I8
            9.I9
            10.I10
            11.I11
            12.I12
            13.I13
            14.C1
            15.C2
            16.C3
            17.C4
            18.C5
            19.C6
            20.C7
            21.C8
            22.C9
            23.C10
            24.C11
            25.C12
            26.C13
            27.C14
            28.C15
            29.C16
            30.C17
            31.C18
            32.C19
            33.C20
            34.C21
            35.C22
            36.C23
            37.C24
            38.C25
            39.C26
        """
        self._num_feats = 39
        self._num_dense_feats = 13
        self._num_sparse_feats = 26
        super(CriteoDataset, self).__init__(data_path=data_path,
                                                        num_feats=self._num_feats,
                                                        num_dense_feats=self._num_dense_feats,
                                                        num_sparse_feats=self._num_sparse_feats,
                                                        repreprocess=repreprocess,
                                                        use_pretrained_embeddings=use_pretrained_embeddings,
                                                        with_head=False,
                                                        logger=logger)

        self._threshold = 10

        self._read()

    def _get_feat_dict(self):
        feat_cnt = defaultdict(lambda: defaultdict(int))
        with open(self._data_path) as f:
            # ignore header
            if self._with_head:
                f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Count feats of Criteo dataset.')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self._num_feats + 1:
                    continue
                # TODO 是否要把 dense 特征变成 sparse 特征
                for i in range(self._num_dense_feats, self._num_feats):
                    feat_cnt[i][values[i + 1]] += 1
        feat_dict = {i: {feat for feat, c in cnt.items() if c >= self._threshold} for i, cnt in feat_cnt.items()}
        feat_dict = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_dict.items()}
        defaults = {i: len(cnt) for i, cnt in feat_dict.items()}

        return feat_dict, defaults

    def _transform_feat(self, feat_dict, defaults):
        dense = []
        sparse = []
        labels = []
        with open(self._data_path) as f:
            # ignore header
            if self._with_head:
                f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Transform feats of Criteo dataset.')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self._num_feats + 1:
                    continue
                record = np.zeros(self._num_feats, dtype=np.int64)
                labels.append(int(values[0]))
                # dense features
                for i in range(0, self._num_dense_feats):
                    # TODO 用 scale 还是 min-max scaler
                    record[i] = scale(values[i + 1])
                # sparse features
                for i in range(self._num_dense_feats, self._num_feats):
                    record[i] = feat_dict[i].get(values[i + 1], defaults[i])
                dense.append(record[:self._num_dense_feats])
                sparse.append(record[self._num_dense_feats:])

        return dense, sparse, labels

    def _read(self):
        if self._repreprocess:
            self._logger.info("Start to preprocess criteo dataset")
            # 0 label, 1-13 dense features, 14-39 sparse features
            feat_dict, defaults = self._get_feat_dict()
            dense_data, sparse_data, labels = self._transform_feat(feat_dict, defaults)

            self._dense_feats = np.array(dense_data, dtype=np.int64)
            self._sparse_feats = np.array(sparse_data, dtype=np.int64)
            self._labels = np.array(labels, dtype=np.int64)
            for i in range(self._num_dense_feats):
                self._field_dims[i] = 1
            for i, f in feat_dict.items():
                self._field_dims[i] = len(np.unique(self._sparse_feats[:, i - self._num_dense_feats]))

            try:
                os.remove(os.path.join(self._save_path, PRE_DENSE_FEATES_FILE))
                os.remove(os.path.join(self._save_path, PRE_SPARSE_FEATES_FILE))
                os.remove(os.path.join(self._save_path, LABELS_FEATES_FILE))
                os.remove(os.path.join(self._save_path, FIELD_DIMS_FILE))
            except Exception as e:
                pass
            np.save(os.path.join(self._save_path, PRE_DENSE_FEATES_FILE), self._dense_feats)
            np.save(os.path.join(self._save_path, PRE_SPARSE_FEATES_FILE), self._sparse_feats)
            np.save(os.path.join(self._save_path, LABELS_FEATES_FILE), self._labels)
            np.save(os.path.join(self._save_path, FIELD_DIMS_FILE), self._field_dims)
            self._split_dataset()
            self._logger.info("Finish preprocess criteo dataset")
        super(CriteoDataset, self)._read()


if __name__ == '__main__':
    data_path = f'{PROJECT_PATH}/codes/datasets/data/criteo.txt'
    logger = get_logger('criteo', 'read_criteo.log')
    ds = CriteoDataset(data_path, repreprocess=True, logger=logger)
    print(f"{ds.labels.shape} {ds.dense_feats.shape} {ds.sparse_feats.shape}")
