import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils

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
    os.system(f'./self-attn qkv_linear {args.input_file} {args.seq_len} {layer.self_attn.num_heads} {layer.self_attn.head_dim} {workdir} {layer_prefix} {args.output_file}')

    Q, K, V = fileio_utils.load_int('temp_Q.bin').reshape(args.seq_len, embed_dim) / (1 << 16), fileio_utils.load_int('temp_K.bin').reshape(args.seq_len, embed_dim) / (1 << 16), fileio_utils.load_int('temp_V.bin').reshape(args.seq_len, embed_dim)

    Q = Q.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    K = K.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    V = V.view(args.seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)

    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(torch.randn(1, args.seq_len, embed_dim, device = 0), torch.arange(args.seq_len, device = 0).unsqueeze(0))

    # print(Q.shape, K.shape, V.shape, cos.shape, sin.shape)

    Q = (Q * cos) + (rotate_half(Q) * sin)
    K = (K * cos) + (rotate_half(K) * sin)

    ref = torch.nn.functional.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(layer.self_attn.head_dim), dim = -1)
    attn_out = []
    for i in range(layer.self_attn.num_heads):
        head_Q, head_K, head_V = Q[i], K[i], V[i]
        fileio_utils.save_int(head_Q, 1 << 16, f'temp_head_Q.bin')
        fileio_utils.save_int(head_K, 1 << 16, f'temp_head_K.bin')
        V[i].cpu().detach().numpy().astype(np.int32).tofile('temp_head_V.bin')
        Vf = V[i] / (1<<16)

        os.system(f'./self-attn head {args.input_file} {args.seq_len} {layer.self_attn.num_heads} {layer.self_attn.head_dim} {workdir} {layer_prefix} {args.output_file}')
        Y = fileio_utils.load_long('temp_head_Y.bin').reshape(args.seq_len, args.seq_len)
        Yf = Y / (1<<40)
        _out = fileio_utils.load_int('temp_head_out.bin')
        attn_out.append(_out)

    attn_out = torch.stack(attn_out).reshape(layer.self_attn.num_heads, args.seq_len, layer.self_attn.head_dim) # num_heads x seq_len x head_dim
    attn_out = attn_out.transpose(0, 1).reshape(args.seq_len, embed_dim)
    print(attn_out, attn_out / (1 << 16), attn_out.shape)
    attn_out.cpu().detach().numpy().astype(np.int32).tofile('temp_attn_out.bin')

    os.system(f'./self-attn o_linear {args.input_file} {args.seq_len} {layer.self_attn.num_heads} {layer.self_attn.head_dim} {workdir} {layer_prefix} {args.output_file}')
    os.system('rm ./temp*.bin')

    # os.system('nvidia-smi')

