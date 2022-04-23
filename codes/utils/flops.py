from thop import profile, clever_format


def get_model_params_flops(model, dataset, batch_size=4096, use_gpu=False, logger=None):
    model.eval()
    dense, sparse, label = dataset.sample_a_batch(batch_size=batch_size)
    if use_gpu:
        dense, sparse, label = dense.cuda(), sparse.cuda(), label.cuda()
    if logger:
        logger.info(f"dense.shape={dense.shape},"
                    f"sparse.shape={sparse.shape},"
                    f"dataset.field_dims={dataset.field_dims}"
                    f"dataset.num_dense_feats={dataset.num_dense_feats}"
                    f"dataset.num_sparse_feats={dataset.num_sparse_feats}")
    macs, params = profile(model, inputs=[(dense, sparse)])

    flops = (macs / (1000 ** 2)) / 2

    return flops