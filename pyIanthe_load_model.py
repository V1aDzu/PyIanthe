import os
import pyIanthe_config

os.environ["HF_HOME"] = pyIanthe_config.HF_HOME
os.environ["HF_DATASETS_CACHE"] = pyIanthe_config.HF_DATASETS_CACHE
os.environ["HF_METRICS_CACHE"] = pyIanthe_config.HF_METRICS_CACHE

from huggingface_hub import snapshot_download

# Визначаємо директорію моделі
author, model_name = pyIanthe_config.MODEL_ID.split("/")
model_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, author, model_name)

# Створюємо папку, якщо її немає
os.makedirs(model_dir, exist_ok=True)

# Перевіряємо, чи вже існує config.json — якщо так, модель вже завантажена
if not os.path.exists(os.path.join(model_dir, "config.json")):
    print("Завантажуємо модель на диск...")
    snapshot_download(
        repo_id=pyIanthe_config.MODEL_ID,
        #cache_dir=model_dir,
        cache_dir=pyIanthe_config.HF_CACHE_DIRNAME,  # використовується для кешування
        local_dir=model_dir
    )
    print("Завантаження завершено.")
else:
    print("Модель вже є на диску:", model_dir)

# Тепер можна завантажувати модель у пам'ять за потреби
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Модель готова до використання.")
