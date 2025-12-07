import os
import sys
import pyIanthe_config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

modules_path = os.path.join(os.path.dirname(__file__), 'modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

author, model_name = pyIanthe_config.MODEL_ID.split("/")

os.makedirs(pyIanthe_config.FOLDER_MODELS, exist_ok=True)
os.makedirs(f"{pyIanthe_config.FOLDER_MODELS}\{author}" , exist_ok=True)

models_path_w_name = f"{pyIanthe_config.FOLDER_MODELS}\{author}\{model_name}"

if not os.path.exists(model_path_w_name):
    # Автозавантаження до папки
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=model_path_w_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=model_path_w_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path_w_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_w_name)

print("Модель була збережена у:", model_path_w_name)
