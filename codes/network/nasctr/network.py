#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.autograd import Variable

from codes.utils.architectures import Architecture
from codes.blocks.block import BLOCKS
from codes.utils.params import init_block2params, get_num_params, set_num_params, get_num_params_info
from codes.utils.utils import NASCTR, get_logger


@lru_cache(maxsize=2)
def get_std(x, std_lambda):
    return std_lambda * torch.std(x).cpu().item()


class MixedBlock(nn.Module):

    def __init__(self, block_params, node2type, block_keys, num_inputs, block_id, num_init_node, std_lambda=0):
        super(MixedBlock, self).__init__()
        self._blocks = nn.ModuleList()
        self._node2type = node2type
        self._num_blocks_type = len(block_keys)
        self._num_inputs = num_inputs
        self._block_id = block_id
        self._num_init_node = num_init_node
        self._std_lambda = std_lambda
        global BLOCKS2PARAMS
        for block in block_keys:
            # TODO block的初始化
            block_params['num_inputs'] = self._num_inputs
            blk = BLOCKS[block](block_params)
            if get_num_params(blk.name) == -1:
                set_num_params(blk.name, blk.num_params)
            self._blocks.append(blk)

    def forward(self, prev_nodes, weights, std_lambda=0):
        # two input
        weights_list = weights.data.cpu().tolist()
        nodes = []
        for i, node in enumerate(prev_nodes):
            if self._block_id >= 2 and i < self._num_init_node and self._std_lambda > 0:
                std = get_std(node, self._std_lambda)
                new_node = node.add(torch.normal(mean=0.0, std=std, size=node.shape, device=node.device))
                nodes.append(new_node)
            else:
                nodes.append(node)

        num_prev_nodes = len(prev_nodes)
        chosen_prev_node1 = weights_list[:num_prev_nodes].index(1)
        if self._num_inputs == 2:
            chosen_prev_node2 = weights_list[num_prev_nodes:num_prev_nodes * 2].index(1)

        inputs = []
        x1 = sum([w * input if w.data > 0 else w for input, w in zip(nodes, weights[:num_prev_nodes])])
        node_type1 = self._node2type[chosen_prev_node1]
        inputs.append((x1, node_type1))
        if self._num_inputs == 2:
            x2 = sum([w * input if w.data > 0 else w for input, w in zip(nodes, weights[num_prev_nodes:num_prev_nodes * 2])])
            node_type2 = self._node2type[chosen_prev_node2]
            inputs.append((x2, node_type2))
        result = [w * blk(inputs) if w.data > 0 else w for w, blk in
                  zip(weights[-self._num_blocks_type:], self._blocks)]

        return sum(result)


class Network(nn.Module):
    def __init__(self, dataset, args, logger=None):
        super(Network, self).__init__()
        self._dataset = dataset
        self._args = args
        if logger is None:
            self._logger = get_logger(NASCTR)
        else:
            self._logger = logger

        init_block2params()

        self._reg = args.weights_reg
        self._num_sparse_feats = dataset.num_sparse_feats
        self._num_dense_feats = dataset.num_dense_feats
        self._field_dims = dataset.field_dims
        self._sparse_offsets = np.array((0, *np.cumsum(self._field_dims)[self._num_dense_feats:-1]), dtype=np.long)

        self._embedding_dim = args.embedding_dim
        self._raw_dense_dim = self._embedding_dim * self._num_dense_feats
        self._raw_sparse_dim = self._embedding_dim * self._num_sparse_feats
        self._block_in_dim = args.block_in_dim
        self._block_out_dim = args.block_in_dim

        self._num_blocks = args.num_block
        self._num_free_block = min(self._num_blocks, args.num_free_block)
        self._max_skip_connect = args.max_skip_connect
        self._num_input_node = 2
        self._num_init_node = 0

        self._raw_dense_linear = None
        self._raw_sparsese_linear = None

        prev_nodes = []
        if self._num_dense_feats > 0:
            prev_nodes.append(1)
            self._num_init_node += 1
        if self._num_sparse_feats > 0:
            prev_nodes.append(2)
            self._num_init_node += 1

        # TODO 是不是每个 block 单独一个 linear
        self._block_params = {
            'raw_dense_linear': self._raw_dense_dim,
            'raw_sparse_linear': self._raw_sparse_dim,
            'block_in_dim': self._block_in_dim,
            'block_out_dim': self._block_out_dim,
            'embedding_dim': self._embedding_dim,
            'num_sparse_feats': self._num_sparse_feats,
            'num_dense_feats': self._num_dense_feats,
            'num_feats': len(self._field_dims),
            'num_inputs': 2
        }
        self._logger.info(f"block_params {self._block_params}")

        if args.block_keys is None:
            self._blocks_keys = BLOCKS.keys()
        else:
            self._blocks_keys = args.block_keys
        self._num_blocks_type = len(self._blocks_keys)

        if args.use_pretrained_embeddings and dataset.pretrained_embeddings is not None:
            self._embedding_table = dataset.pretrained_embeddings
            self._logger.info("Use pretrained embeddings")
        else:
            self._embedding_table = nn.Embedding(sum(self._field_dims), self._embedding_dim)
            torch.nn.init.xavier_uniform_(self._embedding_table.weight.data)
            self._logger.info(f"Initialize the embeddings")

        self._node2type = prev_nodes + [3] * self._num_blocks
        self._blocks = nn.ModuleList()
        self._classifier = None

        self._use_gpu = args.use_gpu

    def get_tag(self):
        return f'nasctr_{self._num_blocks}_{self._num_free_block}_{self._max_skip_connect}_{self._embedding_dim}_{self._block_in_dim}_{self._arch_reg}_{self._reg}'

    def compute_loss(self, predicts, labels, regs, use_reg=True, use_arch_loss=True):
        raise NotImplementedError()


class FixedNetwork(Network):
    def __init__(self, dataset, args, arch, logger=None):
        super(FixedNetwork, self).__init__(dataset, args, logger=logger)

        self._logger.info(f"FixedNetwork: {arch}")
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._block_inputs = []
        for block_type, inputs_idxes in arch.blocks:
            self._block_params['num_inputs'] = len(inputs_idxes)
            self._blocks.append(BLOCKS[block_type](self._block_params))
            self._block_inputs.append(inputs_idxes)
        self._concat_blocks_idxes = list(arch.concat)
        self._classifier = nn.Linear(len(self._concat_blocks_idxes) * self._block_out_dim, 1)

    def forward(self, input):
        raw_dense, raw_sparse = input
        nodes = []
        # norms = []

        if self._num_dense_feats > 0:
            dense_indexes = [i for i in range(self._num_dense_feats)]
            dense_emb = self._embedding_table(raw_dense.new_tensor(dense_indexes))
            dense_emb = dense_emb * raw_dense.unsqueeze(-1)
            nodes.append(dense_emb)
        if self._num_sparse_feats > 0:
            sparse_indexes = raw_sparse + raw_sparse.new_tensor(self._sparse_offsets).unsqueeze(0)
            sparse_emb = self._embedding_table(sparse_indexes)
            nodes.append(sparse_emb)

        for i, block in enumerate(self._blocks):
            if i == 0 and self._num_init_node == 1:
                inputs = [(nodes[self._block_inputs[i][0]], self._node2type[self._block_inputs[i][0]])]
            else:
                inputs = [(nodes[self._block_inputs[i][0]], self._node2type[self._block_inputs[i][0]]),
                          (nodes[self._block_inputs[i][1]], self._node2type[self._block_inputs[i][1]])]
            node = block(inputs)
            nodes.append(node)
        output = torch.cat([nodes[i] for i in self._concat_blocks_idxes], dim=1)
        logits = self._classifier(output)

        norms = sum(torch.norm(p) for name, p in self.named_parameters() if "auxiliary" not in name and p is not None and p.requires_grad)
        regs = self._reg * norms

        return logits, regs

    def compute_loss(self, predicts, labels, regs, use_reg=True, use_arch_loss=False):
        labels = labels.float()

        loss = self._criterion(predicts.squeeze(), labels.squeeze())
        if use_reg:
            loss += regs

        return loss


class DynamicNetwork(Network):
    def __init__(self, dataset, args, logger=None, e_greedy=0):
        super(DynamicNetwork, self).__init__(dataset, args, logger=logger)
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._arch_reg = args.arch_reg
        # self._weight_reg = args.weight_reg
        self._e_greedy = e_greedy
        self._std_lambda = args.std_lambda
        # TODO the value
        self._network_weight_decay = args.arch_weight_decay

        self._num_prev_nodes = [self._num_init_node + i for i in range(self._num_blocks)]

        self._initialize()

        self.saved_params = []
        for w in self._arch_params:
            temp = w.data.clone()
            self.saved_params.append(temp)

        self._classifier = nn.Linear(self._num_blocks * self._block_out_dim, 1)
        self._logger.info(f"The parameter amount of each block: {get_num_params_info()}")

    def new(self):
        model_new = DynamicNetwork(self._dataset, self._args, self._criterion, self._logger).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        raw_dense, raw_sparse = input
        nodes = []
        norms = []
        if self._num_dense_feats > 0:
            dense_indexes = [i for i in range(self._num_dense_feats)]
            dense_emb = self._embedding_table(raw_dense.new_tensor(dense_indexes))
            dense_emb = dense_emb * raw_dense.unsqueeze(-1)
            nodes.append(dense_emb)
            norms.append(torch.norm(dense_emb))
        if self._num_sparse_feats > 0:
            sparse_indexes = raw_sparse + raw_sparse.new_tensor(self._sparse_offsets).unsqueeze(0)
            sparse_emb = self._embedding_table(sparse_indexes)
            nodes.append(sparse_emb)
            norms.append(torch.norm(sparse_emb))

        for i, block in enumerate(self._blocks):
            weights = self._arch_params[i]
            for j in range(self._get_num_input_node(i)):
                assert weights[j * self._num_prev_nodes[i]: (j + 1) * self._num_prev_nodes[i]].sum() == 1
            assert weights[-self._num_blocks_type:].sum() == 1
            node = block(nodes, weights, std_lambda=self._std_lambda)
            nodes.append(node)
        output = torch.cat(nodes[self._num_init_node:], dim=1)
        logits = self._classifier(output)

        regs = self._reg * sum(norms)

        return logits, regs

    def step(self, dense_train, sparse_train, labels_train, dense_valid,
             sparse_valid, labels_valid, arch_optimizer):
        self.zero_grad()
        arch_optimizer.zero_grad()

        # discretization
        self.binarize(self._e_greedy)

        loss = self._backward_step(dense_valid, sparse_valid, labels_valid)
        # restore weight before updating
        self.restore()
        # update the architecture parameters
        arch_optimizer.step()
        return loss

    def _backward_step(self, dense_valid, sparse_valid, labels_valid):
        predicts, regs = self((dense_valid, sparse_valid))
        loss = self.compute_loss(predicts, labels_valid, regs)
        # loss += self._weight_loss()
        loss.backward()
        return loss

    def compute_loss(self, predicts, labels, regs, use_reg=True, use_arch_loss=True):
        labels = labels.float()
        loss = self._criterion(predicts.squeeze(), labels.squeeze())
        if use_reg:
            loss += regs
        if use_arch_loss and self._arch_reg > 0:
            arch_loss = self._arch_loss()
            loss += arch_loss
        return loss

    def _arch_loss(self):
        normal_p = []
        total_params = 0
        for key in self._blocks_keys:
            total_params += get_num_params(key)
        for key in self._blocks_keys:
            normal_p.append(get_num_params(key) / total_params)
        normal_p = torch.Tensor(normal_p)
        if self._use_gpu:
            normal_p = normal_p.cuda()
        normal_p = Variable(normal_p, requires_grad=False)

        regs = [torch.pow(self._arch_params[i][-self._num_blocks_type:], 2) * normal_p for i in range(self._num_blocks)]

        return sum([r.sum() for r in regs]) * self._arch_reg

    def _initialize(self):
        self._arch_params = []
        # TODO 是否要添加一个空的 node

        for i in range(self._num_blocks):
            alphas = torch.ones(self._get_num_input_node(i) * self._num_prev_nodes[i] + self._num_blocks_type) / 2
            alphas = alphas + (torch.randn_like(alphas) * 1e-3)
            if self._use_gpu:
                alphas = alphas.cuda()
            alphas = Variable(alphas, requires_grad=True)
            self._arch_params.append(alphas)

        for i in range(self._num_blocks):
            mixedBlock = MixedBlock(self._block_params, self._node2type, self._blocks_keys,
                                    num_inputs=self._get_num_input_node(i), block_id=i,
                                    num_init_node=self._num_init_node, std_lambda=self._std_lambda)
            self._blocks.append(mixedBlock)

    def save_params(self):
        for index, value in enumerate(self._arch_params):
            self.saved_params[index].copy_(value.data)

    def clip(self):
        m = nn.Hardtanh(0, 1)
        for index in range(len(self._arch_params)):
            self._arch_params[index].data = m(self._arch_params[index].data)

    def _get_num_input_node(self, block_id):
        if block_id == 0 and self._num_init_node == 1:
            return 1
        return self._num_input_node

    def binarize(self, e_greedy=0):
        self.save_params()
        max_skip_connect = self._max_skip_connect
        input_nodes = []
        block_types = []
        prev = [np.array([j * self._num_prev_nodes[i] for j in range(self._get_num_input_node(i))]) for i in range(self._num_blocks)]
        if np.random.rand() <= e_greedy:
            for i in range(self._num_blocks):
                nodes = prev[i] + np.random.choice(self._num_init_node + i, self._get_num_input_node(i), replace=False)
                input_nodes.append(nodes)
                block_types.append(
                    self._get_num_input_node(i) * self._num_prev_nodes[i]
                    + np.random.choice(range(self._num_blocks_type), 1)[0])
        else:
            cnt = 0
            all_inputs = []
            # first two blocks can choose any node as input
            for i in range(self._num_free_block):
                weights = self._arch_params[i]#.data.cpu().numpy()
                inputs = [weights[j * self._num_prev_nodes[i]:(j + 1) * self._num_prev_nodes[i]] for j in range(self._get_num_input_node(i))]
                all_inputs.append(inputs)
                nodes = []
                for input in inputs:
                    node = next(filter(lambda x: x not in nodes, input.argsort(descending=True)))
                    node_type = self._node2type[node]
                    if node_type < 3:
                        cnt += 1
                    nodes.append(node)
                nodes = prev[i] + nodes
                input_nodes.append(nodes)
                block_types.append(self._get_num_input_node(i) * self._num_prev_nodes[i] + weights[-self._num_blocks_type:].argmax())
            max_skip_connect -= cnt
            logits = []
            for i in range(self._num_free_block, self._num_blocks):
                weights = self._arch_params[i]#.data.cpu().numpy()
                inputs = [weights[j * self._num_prev_nodes[i]:(j + 1) * self._num_prev_nodes[i]]
                          for j in range(self._get_num_input_node(i))]
                all_inputs.append(inputs)
                for input in inputs:
                    if self._node2type[input.argmax()] < 3:
                        logits.append(input.max())
            if max_skip_connect > 0:
                selected_logits = sorted(logits)[:max_skip_connect]
            else:
                selected_logits = []
            for i in range(self._num_free_block, self._num_blocks):
                weights = self._arch_params[i]#.data.cpu().numpy()
                inputs = all_inputs[i]
                nodes = []
                for input in inputs:
                    node_iter = iter(filter(lambda x: x not in nodes, input.argsort(descending=True)))
                    while True:
                        node = next(node_iter)
                        node_type = self._node2type[node]
                        if not (node_type < 3 and input[node] not in selected_logits):
                            break
                    nodes.append(node)
                nodes = prev[i] + nodes
                input_nodes.append(nodes)
                block_types.append(self._get_num_input_node(i) * self._num_prev_nodes[i] + weights[-self._num_blocks_type:].argmax())

        maxIndexs = (input_nodes, block_types)
        self.proximal_step(maxIndexs)

    def restore(self):
        for index in range(len(self._arch_params)):
            self._arch_params[index].data = self.saved_params[index]

    def proximal_step(self, max_indexs):
        input_nodes, block_types = max_indexs
        for i in range(self._num_blocks):
            values = torch.zeros_like(self._arch_params[i].data)
            for j in input_nodes[i]:
                values[j] = 1
            values[block_types[i]] = 1
            self._arch_params[i].data = values

    def arch_parameters(self):
        return self._arch_params

    def get_arch(self, use_softmax=True):
        def _parse(weights, primitives):
            arch = []
            arch_p = []
            num_params = 0
            for i in range(self._num_blocks):
                weight = weights[i].data.cpu().numpy()
                inputs = []
                inputs_p = []
                for j in range(self._get_num_input_node(i)):
                    input_p = weight[j * self._num_prev_nodes[i]: (j + 1) * self._num_prev_nodes[i]]
                    node = next(filter(lambda x: x not in inputs, input_p.argsort()[::-1]))
                    inputs.append(node)
                    if use_softmax:
                        input_p = softmax(input_p)
                    inputs_p.append(input_p)
                block_type = primitives[np.argmax(weight[-self._num_blocks_type:])]
                block_type_p = weight[-self._num_blocks_type:]
                if use_softmax:
                    block_type_p = softmax(block_type_p)

                arch.append((block_type, inputs))
                arch_p.append((block_type_p, inputs_p))
                num_params += get_num_params(block_type)

            return arch, arch_p, num_params

        arch, arch_p, params = _parse(self._arch_params, list(self._blocks_keys))

        concat = range(self._num_init_node, self._num_init_node + self._num_blocks)
        arch = Architecture(
            blocks=arch,
            concat=concat
        )
        return arch, arch_p, params
