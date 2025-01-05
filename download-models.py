import os
import sys

model_card = sys.argv[1]
access_token = sys.argv[2]

# download the models
os.environ['TRANSFORMERS_CACHE']="./model-storage"
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(model_card, token = access_token)
    model = AutoModelForCausalLM.from_pretrained(model_card, token = access_token)
except RuntimeError:
    pass