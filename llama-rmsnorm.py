import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for rmsnorm')
parser.add_argument('which', type=str, choices=['input', 'post_attention'], help='To use the input norm or the post-attention norm')
parser.add_argument('seq_len', type=int, help='The sequence length to use for rmsnorm')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for rmsnorm')
parser.add_argument('--output_file', default = 'llama-rmsnorm-output.bin', type=str, help='The output file to use for rmsnorm')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils


if __name__ == '__main__':
    compilation_error = os.system('make rmsnorm')
    if compilation_error:
        print("Error compiling rmsnorm")
        exit(1)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = getattr(model.model.layers[0], f'{args.which}_layernorm')
    # print(layer.eps)
    # print(layer.variance_epsilon)
    # print(layer.weight)
    # for param in layer.parameters():
    #     print(param.shape)
    (embed_dim, ) = layer.weight.shape
    if not os.path.isfile(args.input_file):
        temp_X = torch.randn(args.seq_len, embed_dim, device = 0)
        fileio_utils.save_int(temp_X, 1 << 16, args.input_file)
    X = torch.tensor(np.fromfile(args.input_file, dtype = np.int32).reshape(args.seq_len, embed_dim), device = 0, dtype = float) / (1 << 16)
    rms_inv = 1 / torch.sqrt(torch.mean(X ** 2, dim = 1) + layer.variance_epsilon)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    # print(rms_inv.shape)
    
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    
    os.system(f'./rmsnorm {args.which} {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
    # remove the rms_inv_temp.bin file
    os.remove('rms_inv_temp.bin')
