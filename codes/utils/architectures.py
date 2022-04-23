#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple

Architecture = namedtuple('Architecture', 'blocks concat')


CretioTestNetwork = Architecture(
    blocks=[
        ('MLP-1024', [1, 0]),
        ('ElementWise-min', [1, 0]),
        ('MLP-64', [1, 2]),
        ('MLP-1024', [3, 2]),
        ('Crossnet-1', [2, 0]),
        ('ElementWise-innerproduct', [4, 3]),
        ('ElementWise-min', [5, 6])],
    concat=range(2, 9)
)

AvazuTestNetwork = Architecture(
    blocks=[('ElementWise-innerproduct', [0]),
            ('FM', [0, 1]),
            ('ElementWise-max', [1, 2]),
            ('ElementWise-max', [0, 1]),
            ('MLP-128', [1, 3]),
            ('ElementWise-min', [0, 1]),
            ('Crossnet-4', [1, 6])],
    concat=range(1, 8)
)
