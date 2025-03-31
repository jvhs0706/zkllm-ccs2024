import os
import sys

model_card = sys.argv[1]

# download the models
os.environ['TRANSFORMERS_CACHE']="./model-storage"
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = AutoModelForCausalLM.from_pretrained(model_card)
except RuntimeError:
    pass