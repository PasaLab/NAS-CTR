from codes.utils.architectures import Architecture

CONFIG_TOWER = {
    0: [
        (1, 1, Architecture(
            blocks=[('Crossnet-2', [0]), ('Crossnet-4', [1, 0]), ('Crossnet-3', [1, 2]), ('Crossnet-4', [1, 3]),
                    ('MLP-512', [1, 2]), ('Crossnet-4', [1, 2]), ('MLP-256', [1, 2])], concat=range(1, 8)))
    ]
}

d = {
    'mode': 'nasctr',
    'embedding_dim': 64,
    'block_in_dim': 400,
    'num_block': 7,
    'num_free_block': 20,
    'max_skip_connect': 70,
    'arch_reg': 0.01,
    'use_pretrained_embedding': False,
    'weights_reg': 1e-6,
    'std_lambda': 1
}

CONFIG_CANDIDATES = [
    d
]
