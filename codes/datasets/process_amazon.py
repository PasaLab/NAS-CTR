import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import os
import gzip
import json
import collections
import random
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        # if i >= 10000:
        #     break
    return pd.DataFrame.from_dict(df, orient='index')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


if __name__ == '__main__':
    f1 = '/root/datasets/nasctr/amazon/Movies_and_TV.json.gz'
    f2 = '/root/datasets/nasctr/amazon/meta_Movies_and_TV.json.gz'
    df1 = getDF(f1)
    print(f"read {f1} len={len(df1)}")
    df2 = getDF(f2)
    print(f"read {f2} len={len(df2)}")

    print(f"df1 transform")
    df1 = df1[['reviewerID', 'asin', 'style']]
    df1 = df1.fillna('')
    df1['style'] = df1['style'].map(lambda x: '' if x == '' else x.get('Format:', '').strip())

    print(f"df2 transform")
    df3 = df2[['asin', 'brand', 'price', 'category']]
    df3['price'] = df3['price'].map(lambda x: '' if not x else x.replace('$', '').strip()).to_list()
    df3['price'] = df3['price'].map(lambda x: 0 if not is_number(x) else float(x)).to_list()
    df3['price'] = pd.qcut(df3['price'], q=50, duplicates='drop', labels=False)
    # df3['price'] = df3['price'].map(lambda x: '' if not is_number(x) else str(x)).to_list()
    df3['ctg1'] = df3['category'].map(
        lambda x: str(x).split(',')[1].replace(']', '').replace('\'', '').strip() if len(str(x).split(',')) > 1 else '')
    df3['ctg2'] = df3['category'].map(
        lambda x: str(x).split(',')[2].replace(']', '').replace('\'', '').strip() if len(str(x).split(',')) > 2 else '')
    df2 = df3.drop(['category'], axis=1)

    records = df1.values.tolist()
    products = sorted(df2.values.tolist(), key=lambda x:x[0])

    u2i = collections.defaultdict(set)
    # u2s = collections.defaultdict(list)
    items = set()
    samples = []

    pool = ThreadPool(processes=32)

    n = len(products)
    def binary_search(target):
        l, r = 0, n - 1
        while l <= r:
            m = l + (r - l) // 2
            if products[m][0] == target:
                return m
            elif products[m][0] < target:
                l = m + 1
            else:
                r = m - 1

        return -1

    print(f"add positive sample {len(records)}")
    def add_positive_sample(row):
        try:
            #     print(row['reviewerID'], row['asin'], row['style'])
            temp = [1, row[0], row[1], row[2]]
            target_row = binary_search(row[1])
            if target_row >= 0:
                target_row = products[target_row]
                #     print(target_row['brand'].item(), target_row['price'].item(), target_row['ctg1'].item(), target_row['ctg2'].item())
                temp.extend([target_row[1], target_row[2], target_row[3], target_row[4]])
                samples.append(temp)
                items.add(row[1])
                u2i[row[0]].add(row[1])
                # u2s[row[0]].append(idx)
            else:
                print(f"not find {row[1]}")
            if len(samples) % 100000 == 0:
                print(len(samples))
        except ValueError as e:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            print(row[1])

    pool.map(add_positive_sample, records)
    print(f'get {len(samples)} positive samples')

    n_samples = len(samples)
    print(f"add negative sample {len(u2i.keys())}")
    def add_negative_sample(data):
        key, value = data
        cnt = 0
        tries = 0
        while cnt < 2 and tries < 10000:
            target_row = random.sample(range(n_samples), 1)[0]
            target_row = samples[target_row]
            tries += 1
            if target_row[2] in value:
                #             print(target, key, value)
                continue
            temp = target_row.copy()
            temp[0] = 0
            temp[1] = key

            samples.append(temp)
            cnt += 1
        if len(samples) % 100000 == 0:
            print(len(samples))

    pool.map(add_negative_sample, u2i.items())

    print(f'get {len(samples)} samples')

    if os.path.exists('amazon.txt'):
        os.remove('amazon.txt')
    with open('amazon.txt', 'w+') as f:
        for sample in samples:
            sample = [str(i) for i in sample]
            f.write(','.join(sample)+'\n')
