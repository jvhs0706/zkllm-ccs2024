import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('log_scaling_factor', type=int, help='The log scaling factor to use. Default is 16')

from transformers import AutoTokenizer, AutoModelForCausalLM

def save_weight_int(int_weight: torch.Tensor, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    int_weight.cpu().detach().numpy().astype(np.int32).tofile(path)


if __name__ == '__main__':
    compilation_error = os.system('make commit-param')
    if compilation_error:
        print("Error compiling commit-param")
        exit(1)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
    scaling_factor = 1 << args.log_scaling_factor
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")

    os.makedirs(f"./zkllm-workdir/Llama-2-{args.model_size}b", exist_ok = True)

    for i, layer in enumerate(model.model.layers):
        for j, w in layer.named_parameters():
            if len(w.shape) == 2:
                w_orig = w.float().T
            else:
                w_orig = w.float()
            w_out = torch.round(w_orig * scaling_factor).to(torch.int32)
            print(f'Max difference of Layer {i}, {j}: {((w_out / scaling_factor) - w_orig).abs().max().item()}')
            pp_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/{j}-pp.bin"
            int_bin_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/layer-{i}-{j}-int.bin"
            commitment_path = f"./zkllm-workdir/Llama-2-{args.model_size}b/layer-{i}-{j}-commitment.bin"
            save_weight_int(w_out, int_bin_path)
            if len(w_out.shape) == 2:
                os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} {w_out.shape[1]}')
            else:
                os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} 1')
        