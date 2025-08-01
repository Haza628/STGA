import torch
import numpy as np
import roma
# ghz
################——————————————————————————————————————————————————————————################
def rotmat_to_unitquat(R):
    matrix, batch_shape = roma.internal.flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return roma.internal.unflatten_batch_dims(quat, batch_shape)


def quat_xyzw_to_wxyz(xyzw):
    assert xyzw.shape[-1] == 4
    return torch.cat((xyzw[...,-1,None], xyzw[...,:-1]), dim=-1)

def quat_wxyz_to_xyzw(wxyz):
    assert wxyz.shape[-1] == 4
    return torch.cat((wxyz[...,1:], wxyz[...,0,None]), dim=-1)

def quat_product(p, q):
    vector = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      torch.cross(p[..., :3], q[..., :3], dim=-1))
    last = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], axis=-1)
    return torch.cat((vector, last[...,None]), dim=-1)


def normalize_numpy(x, axis=1, ord=None):
    norms = np.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    return x / norms


def rotmat_to_unitquat_numpy(R):
    matrix = R
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    decision_matrix = np.empty((num_rotations, 4), dtype=matrix.dtype)
    decision_matrix[:, :3] = np.diagonal(matrix, axis1=1, axis2=2)
    decision_matrix[:, -1] = np.sum(decision_matrix[:, :3], axis=1)
    choices = np.argmax(decision_matrix, axis=1)

    quat = np.empty((num_rotations, 4), dtype=matrix.dtype)

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / np.linalg.norm(quat, axis=1)[:, None]
    return quat


def quat_xyzw_to_wxyz_numpy(xyzw):
    assert xyzw.shape[-1] == 4
    return np.concatenate((xyzw[...,-1,None], xyzw[...,:-1]), axis=-1)

def quat_wxyz_to_xyzw_numpy(wxyz):
    assert wxyz.shape[-1] == 4
    return np.concatenate((wxyz[...,1:], wxyz[...,0,None]), axis=-1)

def quat_product_numpy(p, q):
    vector = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      np.cross(p[..., :3], q[..., :3], axis=-1))
    last = p[..., 3] * q[..., 3] - np.sum(p[..., :3] * q[..., :3], axis=-1)
    return np.concatenate((vector, last[..., None]), axis=-1)

################——————————————————————————————————————————————————————————################


def save_pts(obj_path, pts):
    with open(obj_path, 'w') as obj_file:
        for v in pts:
            obj_file.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    




