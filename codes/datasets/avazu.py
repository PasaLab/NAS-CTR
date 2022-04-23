#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')

from codes.utils.utils import PROJECT_PATH, get_logger

from collections import defaultdict

from tqdm import tqdm


import os
import numpy as np

from codes.datasets.dataset import Dataset, PRE_SPARSE_FEATES_FILE, PRE_DENSE_FEATES_FILE, LABELS_FEATES_FILE, \
    FIELD_DIMS_FILE



class AvazuDataset(Dataset):
    """
    Extra preprocess steps:
        1. remove the infrequent features

    Dataset info:
        0.id: ad identifier
        1.click: 0/1 for non-click/click
        2.hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
        3.C1 -- anonymized categorical variable
        4.banner_pos
        5.site_id
        6.site_domain
        7.site_category
        8.app_id
        9.app_domain
        10.app_category
        11.device_id
        12.device_ip
        13.device_model
        14.device_type
        15.device_conn_type
        16.C14
        17.C15
        18.C16
        19.C17
        20.C18
        21.C19
        22.C20
        23.C21
    """

    def __init__(self, data_path, repreprocess=False, use_pretrained_embeddings=True, logger=None):
        self._num_feats = 22
        self._num_dense_feats = 0
        self._num_sparse_feats = 22
        super(AvazuDataset, self).__init__(data_path=data_path,
                                           num_feats=self._num_feats,
                                           num_dense_feats=self._num_dense_feats,
                                           num_sparse_feats=self._num_sparse_feats,
                                           repreprocess=repreprocess,
                                           use_pretrained_embeddings=use_pretrained_embeddings,
                                           with_head=True,
                                           logger=logger)

        self._threshold = 4

        self._read()

    def _get_feat_dict(self):
        feat_cnt = defaultdict(lambda: defaultdict(int))
        with open(self._data_path) as f:
            # ignore header
            if self._with_head:
                f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Count feats of Avazu dataset.')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self._num_feats + 2:
                    continue
                for i in range(0, self._num_feats):
                    feat_cnt[i][values[i + 2]] += 1
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
            pbar.set_description('Transform feats of Avazu dataset.')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self._num_feats + 2:
                    continue
                record = np.zeros(self._num_feats, dtype=np.int64)
                labels.append(int(values[1]))
                for i in range(0, self._num_feats):
                    record[i] = feat_dict[i].get(values[i + 2], defaults[i])
                dense.append([])
                sparse.append(record)

        return dense, sparse, labels

    def _read(self):
        if self._repreprocess:
            self._logger.info("Start to preprocess avazu dataset")
            # 0-id, 1-label, 2-23 category
            feat_dict, defaults = self._get_feat_dict()
            dense_data, sparse_data, labels = self._transform_feat(feat_dict, defaults)

            self._dense_feats = np.array(dense_data, dtype=np.int64)
            self._sparse_feats = np.array(sparse_data, dtype=np.int64)
            self._labels = np.array(labels, dtype=np.int64)
            for i, f in feat_dict.items():
                self._field_dims[i] = len(np.unique(self._sparse_feats[:, i]))

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
            self._logger.info("Finish preprocess avazu dataset")
        super(AvazuDataset, self)._read()


if __name__ == '__main__':
    data_path = f'{PROJECT_PATH}/codes/datasets/data/avazu.txt'
    logger = get_logger('avazu', 'read_avazu.log')
    ds = AvazuDataset(data_path, repreprocess=True, logger=logger)
    print(f"{ds.labels.shape} {ds.dense_feats.shape} {ds.sparse_feats.shape}")
