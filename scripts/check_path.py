import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyIanthe_config

# --- Путь к модели и токенизатору ---
author, model_name = pyIanthe_config.MODEL_ID.split("/")
model_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, author, model_name)

print("Проверяем папку модели:", model_dir)
print("Содержимое папки:", os.listdir(model_dir))

# --- Проверка токенизатора ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    print("Токенизатор загружен успешно.")
    print("Vocab size:", tokenizer.vocab_size)
except Exception as e:
    print("Ошибка загрузки токенизатора:", e)

# --- Проверка модели ---
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, device_map="auto")
    print("Модель загружена успешно.")
except Exception as e:
    print("Ошибка загрузки модели:", e)

# --- Тестовая генерация (очень короткий prompt) ---
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Генерация успешна. Текст:", generated_text)
