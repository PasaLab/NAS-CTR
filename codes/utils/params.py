#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json


def init_block2params():
    global _num_block_params
    _num_block_params = {}


def get_num_params(blk):
    try:
        return _num_block_params[blk]
    except KeyError:
        return -1


def set_num_params(blk, v):
    _num_block_params[blk] = v


def get_num_params_info():
    return json.dumps(_num_block_params, indent=4, sort_keys=True)
