import torch
import numpy as np

def save_int(t: torch.Tensor, scaling_factor: int, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    t_ = torch.round(t * scaling_factor).to(torch.int32)
    t_.cpu().detach().numpy().astype(np.int32).tofile(path)