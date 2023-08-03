import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)
    return features

def normalized_laplacian(adj, sparse=False):
    adj = sp.coo_matrix(adj)

    d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    if not sparse:
        return adj.todense()

    return adj.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
