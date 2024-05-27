import torch
import numpy as np

def save_int(t: torch.Tensor, scaling_factor: int, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    t_ = torch.round(t * scaling_factor).to(torch.int32)
    t_.cpu().detach().numpy().astype(np.int32).tofile(path)

def save_long(t: torch.Tensor, scaling_factor: int, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    t_ = torch.round(t * scaling_factor).to(torch.int64)
    t_.cpu().detach().numpy().astype(np.int64).tofile(path)

def load_int(path, device = 0):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    return torch.from_numpy(np.fromfile(path, dtype=np.int32)).to(device)

def load_long(path, device = 0):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    return torch.from_numpy(np.fromfile(path, dtype=np.int64)).to(device)