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


if __name__ == '__main__':
    compilation_error = os.system('make self-attn')
    if compilation_error:
        print("Error compiling self-attn")
        exit(1)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = model.model.layers[0]
    
    # print(dir(layer.self_attn.rotary_emb))
    # print(layer.self_attn.rotary_emb.cos_cached.shape)
    # print(layer.self_attn.rotary_emb.sin_cached.shape)
    embed_dim = layer.self_attn.q_proj.in_features
    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(torch.randn(1, args.seq_len, embed_dim, device = 0), torch.arange(args.seq_len, device = 0).unsqueeze(0))
    fileio_utils.save_int(cos.view(args.seq_len, 1, layer.self_attn.head_dim).repeat(1, layer.self_attn.num_heads, 1), 1 << 16, 'cos_temp.bin')
    fileio_utils.save_int(sin.view(args.seq_len, 1, layer.self_attn.head_dim).repeat(1, layer.self_attn.num_heads, 1), 1 << 16, 'sin_temp.bin')

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'

    if not os.path.isfile(args.input_file):
        fileio_utils.save_int(torch.randn(args.seq_len, embed_dim, device = 0), 1 << 16, args.input_file)
    
    os.system(f'./self-attn {args.input_file} {args.seq_len} {layer.self_attn.num_heads} {layer.self_attn.head_dim} {workdir} {layer_prefix} {args.output_file}')
    # remove temp files
    os.remove('cos_temp.bin')
    os.remove('sin_temp.bin')


