'''
transform utils
time 2023/7/20
'''
import numpy as np
import torch
import ipdb

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = quat_normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

def quat_unit(a):
    return quat_normalize(a)

def quat_normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_axis(q, axis=0):
    """ ?? """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def batch_get_quaternion(axis_from, axis_to):
    # 计算从起始坐标系旋转到目标坐标系所需要的旋转四元数
    # 计算旋转轴和旋转角度
    # ipdb.set_trace()
    # axis = np.cross(axis_from, axis_to)
    # norm = np.linalg.norm(axis, axis=-1)

    # angle = np.arctan2(norm, np.sum(axis_from * axis_to, axis=-1)
    angle_list = []
    axis_list = []
    batch = axis_from.shape[0]
    for i in range(batch):
        # Calculate the rotation matrix from axis_from to axis_to
        R = np.dot(axis_to[i].T, axis_from[i])

        # Ensure that the matrix is a valid rotation matrix
        # if not np.allclose(np.dot(R.T, R), np.eye(3)):
        #     raise ValueError("The input axes do not form a valid rotation matrix.")

        # Calculate the angle-axis representation
        angle = np.arccos((np.trace(R) - 1) / 2)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        angle_list.append(angle)
        axis_list.append(axis)

    angle = np.stack(angle_list, axis=0)
    axis = np.stack(axis_list, axis=0)
    # ipdb.set_trace()

    angle = torch.tensor(angle, device="cuda:0")
    axis = torch.tensor(axis, device="cuda:0")
    
    # 将旋转轴和旋转角度转换为四元数
    quaternion = quat_from_angle_axis(angle, axis)
    
    return quaternion

def normalize(vector):
    # 计算向量的范数
    norm = np.linalg.norm(vector)
    
    # 如果范数接近零，则返回原始向量
    if norm < 1e-8:
        return vector
    
    # 归一化向量
    normalized_vector = vector / norm
    
    return normalized_vector