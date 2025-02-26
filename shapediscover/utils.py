import numpy as np
import scipy as sp
import torch

# tensor manipulation

def scipy_sparse_matrix_to_torch_sparse(coo_matrix):
    coo_matrix = sp.sparse.coo_matrix(coo_matrix)
    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    torch_indices = torch.LongTensor(indices)
    torch_values = torch.tensor(values, dtype=float)

    return torch.sparse_coo_tensor(
        torch_indices, torch_values, torch.Size(coo_matrix.shape)
    )
