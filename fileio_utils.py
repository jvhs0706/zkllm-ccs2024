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

def to_int64(tensor: torch.Tensor, log_sf: int):
    tensor_ = tensor.to(torch.float64)
    tensor_ = torch.round(tensor_ * (1 << log_sf)).to(torch.int64)
    return tensor_

def to_float(tensor: torch.Tensor, log_sf: int, to_type: torch.dtype = torch.float32):
    tensor_ = (tensor / (1 << log_sf)).to(to_type)
    return tensor_

def rescale(tensor: torch.Tensor, log_sf: int):
    assert tensor.dtype == torch.int64
    tensor_abs = tensor.abs()
    tensor_abs += (1 << (log_sf - 1))
    tensor_abs >>= log_sf
    tensor = tensor.sign() * tensor_abs
    return tensor

# kill ours
def fromto_int64(tensor: torch.Tensor, log_sf: int, float_dtype: torch.dtype = torch.float64):
    return to_float(to_int64(tensor, log_sf), log_sf, torch.float64)

def compare_q(t: torch.Tensor, t_q: torch.Tensor, log_sf: int):
    t_ = to_float(t_q, log_sf, torch.float64)
    return (t - t_).abs().max().item(), (t - t_).abs().mean().item()