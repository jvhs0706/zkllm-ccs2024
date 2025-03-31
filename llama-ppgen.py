import os
import argparse

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('--model-path', type=str, help='The path to the model to use.')
parser.add_argument('--log-off-factor', type=int, default=5, help='The log offset factor to use. Default is 5')

from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    compilation_error = os.system('make ppgen')
    if compilation_error:
        print("Error compiling ppgen")
        exit(1)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only = True, cache_dir = "./model-storage")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only = True, cache_dir = "./model-storage")

    os.makedirs(f"./zkllm-workdir/{args.model_path.replace('/', '--')}", exist_ok = True)

    for (key, w) in model.model.layers[0].named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0]
            pp_size <<= args.log_off_factor
        elif len(w.shape) == 1:
            (pp_size,) = w.shape
        else:
            raise ValueError(f"Unexpected shape {w.shape} for parameter {key}")
        print(f"Generating PP for {key} with size {pp_size}")
        os.system(f"./ppgen {pp_size} ./zkllm-workdir/{args.model_path.replace('/', '--')}/{key}-pp.bin")