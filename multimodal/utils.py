import os
import sys
import h5py
import torch
import numpy as np
import importlib
import random
import shutil
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
# from colors import colors
# colors = np.array(colors, dtype=np.float32)
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from subprocess import call

import ipdb


def force_mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)

def get_RNN_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('RNN_model.' + model_version)

def collate_feats(b):
    return list(zip(*b))

def collate_feats_pass(b):
    return b

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

# def viz_mask(ids):
#     return colors[ids]

def draw_dot(img, xy):
    out = np.array(img, dtype=np.uint8)
    x, y = xy[0], xy[1]
    neighbors = np.array([[0, 0, 0, 1, 1, 1, -1, -1, 1], \
                          [0, 1, -1, 0, 1, -1, 0, 1, -1]], dtype=np.int32)
    for i in range(neighbors.shape[1]):
        nx = x + neighbors[0, i]
        ny = y + neighbors[1, i]
        if nx >= 0 and nx < img.shape[0] and ny >= 0 and ny < img.shape[1]:
            out[nx, ny, 0] = 0
            out[nx, ny, 1] = 0
            out[nx, ny, 2] = 255

    return out

def get_true_false(d):
    d = int(d)
    if d > 0.5:
        return 1
    return 0

def print_true_false(d):
    d = int(d)
    if d > 0.5:
        return 'True'
    return 'False'

def img_resize(data):
    data = np.array(data, dtype=np.float32)
    mini, maxi = np.min(data), np.max(data)
    data -= mini
    data /= maxi - mini
    data = np.array(Image.fromarray((data*255).astype(np.uint8)).resize((224, 224)), dtype=np.float32) / 255
    data *= maxi - mini
    data += mini
    return data

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def export_label(out, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f\n' % (l[i]))

def export_pts_label(out, v, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], l[i]))

def render_pts_label_png(out, v, l):
    export_pts(out+'.pts', v)
    export_label(out+'.label', l)
    export_pts_label(out+'.feats', v, l)
    cmd = 'xvfb-run -a ~/thea/TheaDepsUnix/Source/TheaPrefix/bin/Thea/RenderShape %s.pts -f %s.feats %s.png 448 448 -v 1,0,0,-5,0,0,0,0,1 >> /dev/null' % (out, out, out)
    call(cmd, shell=True)
    # print("calling", cmd)

def export_pts_color_obj(out, v, c):
    with open(out+'.obj', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

def export_pts_color_pts(out, v, c):
    with open(out+'.pts', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, 'checkpt.pth')
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if os.path.exists(filename):
            checkpt = torch.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch

def get_global_position_from_camera(camera, depth, x, y):
    """
    This function is provided only to show how to convert camera observation to world space coordinates.
    It can be removed if not needed.

    camera: an camera agent
    depth: the depth obsrevation
    x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
    """ 
    cm = camera.get_metadata()
    proj, model = cm['projection_matrix'], cm['model_matrix']
    print('proj:', proj)
    print('model:', model)
    w, h = cm['width'], cm['height']

    # get 0 to 1 coordinate for (x, y) coordinates
    xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

    # get 0 to 1 depth value at (x,y)
    zf = depth[int(y), int(x)]

    # get the -1 to 1 (x,y,z) coordinate
    ndc = np.array([xf, yf, zf, 1]) * 2 - 1

    # transform from image space to view space
    v = np.linalg.inv(proj) @ ndc
    v /= v[3]

    # transform from view space to world space
    v = model @ v

    return v

def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta

def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint

def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector

    Args:
        pose: (4, 4) transformation matrix

    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """

    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (
            1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta

# def viz_mask(ids):
#     return colors[ids]

def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x

def get_random_number(l, r):
    return np.random.rand() * (r - l) + l

def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()

def calc_part_motion_degree(part_motion):
    return part_motion * 180.0 / 3.1415926535

def radian2degree(radian):
    return radian * 180.0 / np.pi

def degree2radian(degree):
    return degree / 180.0 * np.pi

def cal_Fscore(pred, labels):
    TP, TN, FN, FP = 0, 0, 0, 0
    TP += ((pred == 1) & (labels == 1)).sum() # predict 1 label 1
    TN += ((pred == 0) & (labels == 0)).sum() # predict 0 label 0
    FN += ((pred == 0) & (labels == 1)).sum() # predict 0 label 1
    FP += ((pred == 1) & (labels == 0)).sum() # predict 1 label 0
    if TP + FP == 0:
        p = 0
    else:
        p = TP / (TP + FP)
    if TP + FN == 0:
        r = 0
    else:
        r = TP / (TP + FN)
    if r + p == 0:
        F1 = 0
    else:
        F1 = 2 * r * p / (r + p)
    acc = (pred == labels).sum() / len(pred)
    # acc = (TP + TN) / (TP + TN + FN + FP)
    # print((pred == labels), len(pred))
    # print(TP, TN, FN, FP)
    # print(F1, p, r, acc)
    # ipdb.set_trace()
    return F1, p, r, acc

def cal_Fscore_number(pred, labels):
    TP, TN, FN, FP = 0, 0, 0, 0
    TP += ((pred == 1) & (labels == 1)).sum() # predict 1 label 1
    TN += ((pred == 0) & (labels == 0)).sum() # predict 0 label 0
    FN += ((pred == 0) & (labels == 1)).sum() # predict 0 label 1
    FP += ((pred == 1) & (labels == 0)).sum() # predict 1 label 0
    acc_num = (pred == labels).sum()
    all_num = pred.shape[0]
    return TP, TN, FN, FP, acc_num, all_num


class CriticReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, out_info):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = out_info
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # print('buffer: ', self.buffer[0])
        # out_info = map(np.stack, zip(*batch))
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return batch

    def __len__(self):
        return len(self.buffer)
