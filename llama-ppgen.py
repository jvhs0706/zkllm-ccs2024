import os, sys
import argparse

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argumetn('--log_off_factor', type=int, default=0, help='The log offset factor to use. Default is 0')

from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    os.system('make ppgen')
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")

    os.makedirs(f"./zkllm-workdir/Llama-2-{args.model_size}b", exist_ok = True)

    for (i, w) in model.model.layers[0].named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0]
            pp_size <<= args.log_off_factor
        elif len(w.shape) == 1:
            (pp_size,) = w.shape
        else:
            raise ValueError(f"Unexpected shape {w.shape} for parameter {i}")
        
        os.system(f'./ppgen {pp_size} ./zkllm-workdir/Llama-2-{args.model_size}b/{i}-pp.bin')