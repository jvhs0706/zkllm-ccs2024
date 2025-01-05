import os, sys
import argparse
import torch
import numpy as np
import math

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')

from transformers import AutoTokenizer, AutoModelForCausalLM
from fileio_utils import *

VALUE_LOGSF = 16
ACCU_LOGSF = 20

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

if __name__ == '__main__':
    compilation_error = os.system('make self-attn')
    if compilation_error:
        print("Error compiling self-attn")
        exit(1)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = model.model.layers[args.layer]
    embed_dim = layer.self_attn.q_proj.in_features

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    os.system(f'./self-attn linear {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')

    Q, K, V = load_int('temp_Q.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_K.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_V.bin').reshape(args.seq_len, embed_dim) / (1 << 16)

    Q = Q.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    K = K.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    V = V.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)

    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(torch.randn(1, args.seq_len, embed_dim, device = 0), torch.arange(args.seq_len, device = 0).unsqueeze(0))

    Q, K = Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
    Q, K = Q.to(torch.float64), K.to(torch.float64)
    
    A_ = Q @ K.transpose(-2, -1)
    A = to_int64(A_, VALUE_LOGSF)

    # an upper triangular mask for perplexity
    mask = torch.triu(torch.ones(args.seq_len, args.seq_len, device = 0, dtype = bool), diagonal = 1)

    A -= torch.max(A * ~mask, dim = -1, keepdim = True).values 

    shift = math.sqrt(layer.self_attn.head_dim) * torch.log((torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(layer.self_attn.head_dim))) * ~mask).sum(axis = -1, keepdim = True))
    shift = to_int64(shift, ACCU_LOGSF)
    A -= shift
    attn_output = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(layer.self_attn.head_dim)).float()) * ~mask

    attn_output = attn_output @ V
    attn_output = fromto_int64(attn_output, VALUE_LOGSF)

    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output = attn_output.view(args.seq_len, embed_dim)
    attn_output = attn_output.transpose(0, 1).reshape(args.seq_len, embed_dim)
    save_int(attn_output, 1 << 16, 'temp_attn_out.bin') 
    os.system(f'./self-attn attn {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
    os.system('rm ./temp*.bin')

