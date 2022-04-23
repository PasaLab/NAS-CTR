#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-09-16

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils.utils import NASCTR, get_logger

BLOCKS = {
    'FM':
        lambda params: FM('FM', params),
    'MLP-32':
        lambda params: MLP('MLP-32', 32, params),
    'MLP-64':
        lambda params: MLP('MLP-64', 64, params),
    'MLP-128':
        lambda params: MLP('MLP-128', 128, params),
    'MLP-256':
        lambda params: MLP('MLP-256', 256, params),
    'MLP-512':
        lambda params: MLP('MLP-512', 512, params),
    'MLP-1024':
        lambda params: MLP('MLP-1024', 1024, params),
    'ElementWise-sum':
        lambda params: ElementWise('ElementWise-sum', 'sum', params),
    'ElementWise-avg':
        lambda params: ElementWise('ElementWise-avg', 'avg', params),
    'ElementWise-min':
        lambda params: ElementWise('ElementWise-min', 'min', params),
    'ElementWise-max':
        lambda params: ElementWise('ElementWise-max', 'max', params),
    'ElementWise-innerproduct':
        lambda params: ElementWise('ElementWise-innerproduct', 'innerproduct', params),
    'Crossnet-1':
        lambda params: CrossNet('Crossnet-1', 1, params),
    'Crossnet-2':
        lambda params: CrossNet('Crossnet-2', 2, params),
    'Crossnet-3':
        lambda params: CrossNet('Crossnet-3', 3, params),
    'Crossnet-4':
        lambda params: CrossNet('Crossnet-4', 4, params),
    'Crossnet-5':
        lambda params: CrossNet('Crossnet-5', 5, params),
    'Crossnet-6':
        lambda params: CrossNet('Crossnet-6', 6, params),
}

LOGGER = get_logger(NASCTR)


class Block(nn.Module):
    """
    The input shape of raw sparse feature is (batch_size, field_size, embedding_dim).
    The input shape of raw dense feature is (batch_size, field_size, embedding_dim).
    The input shape of inner block is (batch_size, features_size).
    """

    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True, use_dropout=True, dropout_rate=0.5, use_linear=True):
        super(Block, self).__init__()
        self._block_name = block_name
        self._block_in_dim = params['block_in_dim']
        self._block_out_dim = params['block_out_dim']
        self._embedding_dim = params['embedding_dim']
        self._num_sparse_feats = params['num_sparse_feats']
        self._num_dense_feats = params['num_dense_feats']
        self._num_feats = params['num_feats']
        self._num_inputs = params['num_inputs']
        if use_linear:
            if self._num_dense_feats > 0:
                self._raw_dense_linear = nn.Linear(params['raw_dense_linear'], self._block_out_dim)
            if self._num_sparse_feats > 0:
                # LOGGER.info(f"##### params['raw_sparse_linear'] {params['raw_sparse_linear']}")
                self._raw_sparse_linear = nn.Linear(params['raw_sparse_linear'], self._block_out_dim)

        self._use_batchnorm = use_batchnorm
        self._use_relu = use_relu
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate

        self._relu = nn.ReLU()
        self._batchnorm = nn.BatchNorm1d(self._block_out_dim)
        self._dropout = nn.Dropout(self._dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: list, e.g. [(x1, input_type1), (x2, input_type2)]
        input_type == 0 means empty
        input_type == 1 means raw dense features
        input_type == 2 means raw sparse features
        input_type == 3 means inner block output features
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._block_name

    @property
    def num_params(self):
        return self._num_params

    def _count_params(self):
        num_params = sum([p.numel() for p in self.parameters() if p is not None and p.requires_grad])
        # if self._raw_dense_linear:
        #     num_params -= sum([p.numel() for p in self._raw_dense_linear.parameters() if p.requires_grad])
        # if self._raw_sparse_linear:
        #     num_params -= sum([p.numel() for p in self._raw_sparse_linear.parameters() if p.requires_grad])

        return num_params


# TODO 注意 bias 的使用
# TODO 是否要为各种不同的输入定制唯一的 dense 层
# TODO FM 的一阶部分


class FM(Block):
    """
    This block applies FM. The 2-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True, use_dropout=True, dropout_rate=0.5):
        super(FM, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                 use_dropout=use_dropout, dropout_rate=dropout_rate, use_linear=False)
        self._emb_linear = nn.Linear(self._block_in_dim, self._embedding_dim)
        self._output_linear = nn.Linear(self._embedding_dim, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        # none_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                # none_list.append(x)
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            # if input_type == 3 and len(list(filter(lambda i: i[1] in [1, 2], inputs))) > 0:
            if input_type == 3:
                x = self._emb_linear(x)
            if len(x.shape) == 2:
                x = torch.unsqueeze(x, dim=1)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)
        second_order = torch.sub(sum_squared, squared_sum)
        final = 0.5 * second_order
        # final = 0.5 * torch.sum(second_order, dim=1, keepdim=True)
        output = self._output_linear(final)
        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        # if torch.isnan(output).any():
        #     LOGGER.info(f"{self._block_name} find NaN")

        return output


class MLP(Block):
    """
    This block applies MLP. The 3-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, hidden_size, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(MLP, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                  use_dropout=use_dropout, dropout_rate=dropout_rate)
        self._hidden_size = hidden_size
        # self._expand_linear = nn.Linear(self._block_in_dim, self._block_in_dim * self._num_inputs)
        self._hidden_linear = nn.Linear(self._block_in_dim * self._num_inputs, hidden_size)
        self._output_linear = nn.Linear(hidden_size, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                # LOGGER.info(f"#### x.shape{x.shape}")
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        # if len(x_list) == 1:
        #     x = self._expand_linear(x_list[0])
        # else:
        #     x = torch.cat(x_list, dim=1)
        x = torch.cat(x_list, dim=1)

        final = self._hidden_linear(x)
        output = self._output_linear(final)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        # if torch.isnan(output).any():
        #     LOGGER.info(f"{self._block_name} find NaN")

        return output


class ElementWise(Block):
    """
    This block applies inner product. The 3-D array will be converted into 3-D array.
    The elementwise type should be avg, sum, min, max or innerproduct.
    """

    def __init__(self, block_name, elementwise_type, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(ElementWise, self).__init__(block_name, params,
                                          use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                          dropout_rate=dropout_rate)
        self._elementwise_type = elementwise_type
        # self._expand_linear = nn.Linear(self._block_in_dim, self._block_in_dim * self._num_inputs)
        # self._output_linear = nn.Linear(self._block_in_dim * self._num_inputs, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                # print(f"===== dense {x.shape}")
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                # LOGGER.info(f"#### x.shape{x.shape}")
                # print(f"===== sparse {x.shape}")
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        if len(x_list) == 1:
            # x = self._expand_linear(x_list[0])
            # just have one input, don't need to do element-wise operation
            return x_list[0]
        else:
            x = torch.stack(x_list, dim=0)

        if self._elementwise_type == 'avg':
            final = torch.mean(x, dim=0)
        elif self._elementwise_type == 'sum':
            final = torch.sum(x, dim=0)
        elif self._elementwise_type == 'min':
            final, _ = torch.min(x, dim=0)
        elif self._elementwise_type == 'max':
            final, _ = torch.max(x, dim=0)
        elif self._elementwise_type == 'innerproduct':
            final = torch.prod(x, dim=0)
        else:
            final = torch.sum(x, dim=0)

        # output = self._output_linear(final)
        output = final

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        # if torch.isnan(output).any():
        #     LOGGER.info(f"{self._block_name} find NaN")

        return output


class CrossNet(Block):
    """
    This block applies CrossNet. The 3-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, layer_num, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(CrossNet, self).__init__(block_name, params,
                                       use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                       dropout_rate=dropout_rate)
        self._layer_num = layer_num
        # self._expand_linear = nn.Linear(self._block_in_dim, self._block_in_dim * self._num_inputs)
        # self._w = torch.nn.ModuleList([
        #     nn.Linear(self._num_inputs * self._block_in_dim, 1, bias=False) for _ in range(self._layer_num)
        # ])
        # self._b = torch.nn.ParameterList([
        #     nn.Parameter(torch.zeros((self._num_inputs * self._block_in_dim,))) for _ in range(self._layer_num)
        # ])
        self._w = nn.Parameter(torch.randn(self._layer_num, self._block_in_dim * self._num_inputs))
        nn.init.xavier_uniform_(self._w)
        self._b = nn.Parameter(torch.randn(self._layer_num, self._block_in_dim * self._num_inputs))
        nn.init.zeros_(self._b)
        self._bn_list = nn.ModuleList()
        for _ in range(self._layer_num + 1):
            self._bn_list.append(nn.BatchNorm1d(self._block_in_dim * self._num_inputs))
        self._output_linear = nn.Linear(self._block_in_dim * self._num_inputs, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        # if len(x_list) == 1:
        #     x = self._expand_linear(x_list[0])
        # else:
        #     x = torch.cat(x_list, dim=1)
        x = torch.cat(x_list, dim=1)

        x = self._bn_list[0](x)
        x0 = x
        for i in range(self._layer_num):
            # LOGGER.info(f"{self._block_name} before {i} min {x.min()} max {x.max()}")
            # xw = self._w[i](x)
            # x = xw * x0 + x + self._b[i]
            w = torch.unsqueeze(self._w[i, :].T, dim=1)                  # In * 1
            xw = torch.mm(x, w)                                       # None * 1
            x = torch.mul(x0, xw) + self._b[i, :] + x   # None * In
            # LOGGER.info(f"{self._block_name} before bn {i} min {x.min()} max {x.max()}")
            x = self._bn_list[i + 1](x)
            # LOGGER.info(f"{self._block_name} after {i} min {x.min()} max {x.max()}")

        output = self._output_linear(x)
        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        # if torch.isnan(output).any():
        #     LOGGER.info(f"{self._block_name} find NaN")

        return output


class CIN(Block):
    def __init__(self, block_name, cross_layer_num, params, split_half=True, use_batchnorm=True, use_relu=True,
                 use_dropout=True, dropout_rate=0.5):
        super(CIN, self).__init__(block_name, params,
                                  use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                  dropout_rate=dropout_rate)
        self._num_layers = cross_layer_num
        self._cross_layer_dims = [400] * cross_layer_num
        self._split_half = split_half
        self._conv_layers = torch.nn.ModuleList()
        input_dim, prev_dim, final_input_dim = self._num_feats, self._num_feats, 0
        for i in range(self._num_layers):
            cross_layer_size = self._cross_layer_dims[i]
            self._conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1, stride=1, bias=True))
            if self._split_half and i != self._num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            final_input_dim += prev_dim
        self._output_linear = torch.nn.Linear(final_input_dim, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        for x, input_type in inputs:
            if input_type == 1 or input_type == 2:
                x_list.append(x)
            else:
                raise Exception(f"{self._block_name} not support this input type {input_type}.")
        x = torch.stack(x_list, dim=1)
        xs = list()
        x0, h = x.unsqueeze(2), x  # (B, F0, 1, E) (B, F0, E)
        for i in range(self._num_layers):
            x = x0 * h.unsqueeze(1)  # (B, F0, 1, E) X (B, 1, F1, E) => (B, F0, F1, E)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.reshape(batch_size, f0_dim * fin_dim, embed_dim)  # (B, F0 * F1, E)
            x = F.relu(self._conv_layers[i](x))  # (B, F2, E)
            if self._split_half and i != self._num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        output = torch.sum(torch.cat(xs, dim=1), 2)
        output = self._output_linear(output)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class InnerProduct(Block):
    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True,
                 use_dropout=True, dropout_rate=0.5):
        super(InnerProduct, self).__init__(block_name, params,
                                  use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                  dropout_rate=dropout_rate)
        # TODO block out dim 是否要减半
        # linear parts
        linear_weights = torch.randn((self._block_out_dim // 2, self._num_fields, self._embedding_dim))
        nn.init.xavier_uniform_(linear_weights)
        self._linear_weights = nn.Parameter(linear_weights)
        # product parts
        theta = torch.randn((self._block_out_dim // 2, self._num_fields))
        nn.init.xavier_uniform_(theta)
        self._weights = nn.Parameter(theta)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        for x, input_type in inputs:
            if input_type == 1 or input_type == 2:
                x_list.append(x)
            else:
                raise Exception(f"{self._block_name} not support this input type {input_type}.")
        x = torch.stack(x_list, dim=1)

        lz = torch.einsum('bnm,dnm->bd', x, self._linear_weights)  # (B, D1)
        delta = torch.einsum('bnm,dn->bdnm', x, self._weights)  # (B, D1, N, M)
        lp = torch.einsum('bdnm,bdnm->bd', delta, delta)  # (B, D1)
        output = torch.cat((lz, lp), dim=1)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class OuterProduct(Block):
    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True,
                 use_dropout=True, dropout_rate=0.5):
        super(OuterProduct, self).__init__(block_name, params,
                                           use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                           dropout_rate=dropout_rate)
        # linear parts
        linear_weights = torch.randn((self._block_out_dim // 2, self._num_fields, self._embedding_dim))
        nn.init.xavier_uniform_(linear_weights)
        self._linear_weights = nn.Parameter(linear_weights)
        # product parts
        quadratic_weights = torch.randn((self._block_out_dim // 2, self._embedding_dim, self._embedding_dim))
        nn.init.xavier_uniform_(quadratic_weights)
        self._weights = nn.Parameter(quadratic_weights)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        for x, input_type in inputs:
            if input_type == 1 or input_type == 2:
                x_list.append(x)
            else:
                raise Exception(f"{self._block_name} not support this input type {input_type}.")
        x = torch.stack(x_list, dim=1)

        lz = torch.einsum('bnm,dnm->bd', x, self._linear_weights)  # (B, D1)
        embed_sum = torch.sum(x, dim=1)  # (B, M)
        p = torch.einsum('bm,bn->bmn', embed_sum, embed_sum)  # (B, M, M)
        lp = torch.einsum('bmn,dmn->bd', p, self._weights)  # (B, D1)
        output = torch.cat((lz, lp), dim=1)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output
