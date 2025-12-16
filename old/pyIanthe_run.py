import os
import sys
import pyIanthe_config
from transformers import AutoModelForCausalLM, AutoTokenizer

modules_path = os.path.join(os.path.dirname(__file__), 'modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

model_name = "distilgpt2"  # маленькая модель

os.makedirs(pyIanthe_config.FOLDER_MODEL, exist_ok=True)
model_path_w_name = os.path.join(pyIanthe_config.FOLDER_MODEL, model_name)

if not os.path.exists(model_path_w_name):
    # Автозагрузка модели в папку проекта
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path_w_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path_w_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path_w_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_w_name)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=20)
print(tokenizer.decode(output[0]))
