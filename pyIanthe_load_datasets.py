import os
import pyIanthe_config
# ---------------------------
# Настройки окружения HF
# ---------------------------
os.environ["HF_HOME"] = pyIanthe_config.HF_HOME
os.environ["HF_DATASETS_CACHE"] = pyIanthe_config.HF_DATASETS_CACHE
os.environ["HF_METRICS_CACHE"] = pyIanthe_config.HF_METRICS_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = pyIanthe_config.HF_HUB_WARN_DISABLE
os.environ["TRANSFORMERS_CACHE"] = pyIanthe_config.TF_CACHE_DIRNAME

import re
import json
import time
import shutil
from itertools import islice
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value
from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder

TARGET_FEATURES = Features({
    "text": Value("string")
})

# Перевірка (можна прибрати після налаштування)
print("="*60)
print("ПЕРЕВІРКА ЗМІННИХ СЕРЕДОВИЩА:")
print(f"HF_HOME: {os.environ.get('HF_HOME', '❌ НЕ ВСТАНОВЛЕНО!')}")
print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '❌ НЕ ВСТАНОВЛЕНО!')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '❌ НЕ ВСТАНОВЛЕНО!')}")
print(f"HF_HUB_DISABLE_SYMLINKS_WARNING: {os.environ.get('HF_HUB_DISABLE_SYMLINKS_WARNING', '❌ НЕ ВСТАНОВЛЕНО!')}")
print("="*60)

# ---------------------------
# Каталоги
# ---------------------------
os.makedirs(pyIanthe_config.FOLDER_CORPUS, exist_ok=True)
os.makedirs(pyIanthe_config.FOLDER_TRAIN_DATASET, exist_ok=True)
os.makedirs(getattr(pyIanthe_config, "FOLDER_EVAL_DATASET", "dataset/eval"), exist_ok=True)

SEED = 42  # фиксированный seed для shuffle

def normalize_dataset(ds):
    # 1. если есть поле text — оставляем только его
    if "text" in ds.features:
        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    else:
        raise ValueError("Dataset does not contain 'text' column")

    # 2. приводим features
    ds = ds.cast(TARGET_FEATURES)
    return ds

# ---------------------------
# Утилита: получить имя сплита
# ---------------------------
def extract_split_name(split_string):
    if not isinstance(split_string, str):
        return "train"
    match = re.match(r"^([a-zA-Z_]+)", split_string)
    if match:
        return match.group(1)
    return "train"

# ---------------------------
# Загрузка метаданных
# ---------------------------
corpus_info_path = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.CORPUS_DATA_FILENAME)
with open(corpus_info_path, "r", encoding="utf-8") as f:
    dataset_info = json.load(f)

loaded_datasets = []

# ---------------------------
# Обработка каждого датасета
# ---------------------------
for info in dataset_info:
    dataset_name = info.get("name", "UNKNOWN")
    local_path = os.path.join(pyIanthe_config.FOLDER_CORPUS, dataset_name)
    ds = None
    hf_total = None
    split_name = extract_split_name(info.get("split", "train"))

    print(f"\n[INFO] Обробка датасету: {dataset_name}")

    # --- Проверка локальной копии ---
    local_count = 0
    if os.path.exists(local_path):
        try:
            ds = load_from_disk(local_path)
            local_count = len(ds)
            print(f"[INFO] Локальна копія знайдена: {local_path}, записів: {local_count}")
        except Exception as e:
            print(f"[WARNING] Не вдалося завантажити локальну копію: {e}")
            ds = None
            local_count = 0

        # Проверка доступного количества на HF
        if "hf_id" in info and info["hf_id"]:
            try:                
                config = info.get("config", None)
                if config:
                    builder = load_dataset_builder(info["hf_id"], config)
                else:
                    builder = load_dataset_builder(info["hf_id"])

                if split_name in builder.info.splits:
                    hf_total = builder.info.splits[split_name].num_examples
                    print(f"[INFO] Доступно записів у спліті: '{split_name}' всього на HF: {hf_total}")
                    print(f"[INFO] Опис датасету: {info.get('info', 'Немає опису')}")
                    print(f"[INFO] Якість тексту: {info.get('clean', 'невідома')} Мова: {info.get('lang', 'різні')}")
            except Exception:
                print(f"[WARNING] Не вдалося визначити максимальну кількість записів на HF")

        prompt_msg = f"[PROMPT] Доступні опції: (d) Заменить, (k) Пропустити, (r) Удалить"
        action = input(f"{prompt_msg}: ").strip().lower()

        if action == "k":
            loaded_datasets.append({
                    "dataset": ds,
                    "local": True,
                    "name": dataset_name,
                    "lang": info.get("lang", "різні"),
                    "quality": info.get("clean", "невідомо"),
                    "count": len(ds)
                })
            print(f"[INFO] Пропускаємо {dataset_name}")
            continue

        elif action == "r":
            try:
                timestamp = int(time.time())
                backup_path = f"{local_path}_TO_DELETE_{timestamp}"
                os.rename(local_path, backup_path)
                print(f"[INFO] Старий датасет перейменовано для видалення: {backup_path}")
            except Exception as e:
                print(f"[WARNING] Не вдалося перейменувати старий датасет: {e}")
            continue

        elif action == "d":
            try:
                timestamp = int(time.time())
                backup_path = f"{local_path}_TO_DELETE_{timestamp}"
                os.rename(local_path, backup_path)
                print(f"[INFO] Старий датасет перейменовано для заміни: {backup_path}")
            except Exception as e:
                print(f"[WARNING] Не вдалося перейменувати старий датасет: {e}")

    # --- Если нет локальной копии, проверяем HF ---
    else:
        if "hf_id" not in info or not info["hf_id"]:
            print(f"[INFO] Для {dataset_name} не задан hf_id і локальна копія відсутня, пропускаємо")
            continue

        # Узнаем количество на HF
        try:
            # Поддержка config
            config = info.get("config", None)
            if config:
                builder = load_dataset_builder(info["hf_id"], config)
            else:
                builder = load_dataset_builder(info["hf_id"])
            
            if split_name in builder.info.splits:
                hf_total = builder.info.splits[split_name].num_examples
                print(f"[INFO] Доступно записів у спліті: '{split_name}' всього на HF: {hf_total}")
                print(f"[INFO] Опис датасету: {info.get('info', 'Немає опису')}")
                print(f"[INFO] Якість тексту: {info.get('clean', 'невідома')} Мова: {info.get('lang', 'різні')}")
        except Exception:
            print(f"[WARNING] Не вдалося визначити максимальну кількість записів на HF")

        prompt_msg = "[PROMPT] Доступні опції: (d) Скачать, (k) Пропустити"
        action = input(f"{prompt_msg}: ").strip().lower()

        if action != "d":
            print(f"[INFO] Пропускаємо {dataset_name}")
            continue

    # --- Ввод количества записей (ОБЯЗАТЕЛЬНЫЙ) ---
    while True:
        n_str = input("[PROMPT] Введіть кількість записів для завантаження: ").strip()
        try:
            n = int(n_str)
            if n > 0:
                break
        except ValueError:
            pass
        print("[ERROR] Потрібно ввести додатне ціле число")

    # --- Скачивание нового датасета (STREAMING ONLY) ---
    try:
        print("[INFO] Використовується streaming-режим")

        ds_stream = load_dataset(
            info["hf_id"],
            info.get("config"),
            split=split_name,
            streaming=True
        )

        # Проверяем, что датасет реально читается
        iterator = iter(ds_stream)
        try:
            first = next(iterator)
        except StopIteration:
            print("[ERROR] Датасет існує, але split порожній")
            continue

        # Берем n записей
        ds = Dataset.from_list(list(islice(ds_stream, n)))

        # Фильтруем испорченные/пустые строки
        ds = ds.filter(lambda x: x.get("text") is not None and x["text"].strip() != "")

        try:
            ds = normalize_dataset(ds)
        except Exception as e:
            print(f"[ERROR] Датасет {dataset_name} несумісний для LLM: {e}")
            continue

        ds.save_to_disk(local_path)
        print(f"[INFO] Датасет {dataset_name} завантажено (streaming), записів: {len(ds)}")

        loaded_datasets.append({
            "dataset": ds,
            "local": True,
            "name": dataset_name,
            "lang": info.get("lang", "різні"),
            "quality": info.get("clean", "невідомо"),
            "count": len(ds)
        })        

    except Exception as e:
        print(f"[ERROR] Не вдалося скачати {dataset_name}: {e}")
        continue

# ---------------------------
# Объединение всех загруженных датасетов
# ---------------------------
if not loaded_datasets:
    print("[WARNING] Жоден датасет не було завантажено або знайдено локально. Завершення роботи.")
    raise SystemExit(1)

print(f"[INFO] Готово до об'єднання датасетів: {len(loaded_datasets)}")
for i, d in enumerate(loaded_datasets):
    try:
        loaded_datasets[i]["dataset"] = normalize_dataset(d["dataset"])
    except Exception as e:
        print(f"[WARNING] Не вдалося нормалізувати датасет {i}: {e}")

# Заголовок таблицы
print(f"{'[INFO]  № ':<3}| {'Name':<25} | {'Lang':<10} | {'Quality':<10} | {'Count':>7} |наявність|")
print("-" * 86)

# Вывод всех датасетов
for i, d in enumerate(loaded_datasets, start=1):
    if "text" in d["dataset"].column_names:
        name = d.get("name", "невідомо")
        lang = d.get("lang", "різні")
        quality = d.get("quality", "невідомо")
        count = len(d["dataset"])
        print(f"[INFO] {i:<3}| {name:<25} | {lang:<10} | {quality:<10} | {count:>7} |  доданe  |")
    else:
        name = d.get("name", "невідомо")
        lang = d.get("lang", "невідомо")
        quality = d.get("quality", "невідомо")
        count = len(d["dataset"])
        print(f"[WARN] {i:<3}| {name:<25} | {lang:<10} | {quality:<10} | {count:>7} |ігноровано|")

# Суммарный размер
total_combined = sum(len(d["dataset"]) for d in loaded_datasets if "text" in d["dataset"].column_names)
print("-" * 86)
print(f"{'TOTAL':<66} | {total_combined:>7}")

datasets_to_concat = []
for d in loaded_datasets:
    if "text" in d["dataset"].column_names:
        datasets_to_concat.append(d["dataset"])
    else:
        print(f"[WARNING] Пропускаємо датасет без text:  {d.get('name', 'невідомо')}")

combined_dataset = concatenate_datasets(datasets_to_concat)
combined_dataset = combined_dataset.filter(lambda x: x["text"] not in [None, ""])
total_combined = len(combined_dataset)
print(f"[INFO] Загальний розмір об'єднаного датасету: {total_combined} записів")

# ---------------------------
# Перемешивание и деление на train/eval
# ---------------------------
eval_percent = getattr(pyIanthe_config, "EVAL_PERCENT", 5)
eval_fraction = float(eval_percent) / 100.0
eval_size = max(int(total_combined * eval_fraction), 1)

shuffled = combined_dataset.shuffle(seed=SEED)
eval_dataset = shuffled.select(range(eval_size))
train_dataset = shuffled.select(range(eval_size, total_combined))

train_dataset.save_to_disk(pyIanthe_config.FOLDER_TRAIN_DATASET)
eval_dataset.save_to_disk(pyIanthe_config.FOLDER_EVAL_DATASET)
print(f"[INFO] Train збережено: {pyIanthe_config.FOLDER_TRAIN_DATASET}, записів: {len(train_dataset)}")
print(f"[INFO] Eval збережено: {pyIanthe_config.FOLDER_EVAL_DATASET}, записів: {len(eval_dataset)}")

# ---------------------------
# Проверка токенизатора
# ---------------------------
tokenizer_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, "Qwen", "Qwen2.5-0.5B-Instruct")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    print("[INFO] Токенізатор завантажено успішно.")
    print(f"[INFO] Розмір словника: {tokenizer.vocab_size}")

    sample_count = min(5, len(train_dataset))
    
    sample_texts = [train_dataset[i]["text"] for i in range(sample_count)]    

    if sample_texts:
        tokenized_samples = tokenizer(sample_texts, truncation=True, padding=True)
        print("[INFO] Токенізація тестових записів пройшла успішно.")
        for i, t in enumerate(sample_texts):
            snippet = t[:100].replace("\n", " ")
            ids = tokenized_samples["input_ids"][i][:10]
            print(f"\n[Sample {i+1}] оригінал: {snippet}...")
            print(f"[Sample {i+1}] токени: {ids} ...")
    else:
        print("[INFO] Train-набір занадто малий для тестової токенізації.")
except Exception as e:
    print(f"[ERROR] Помилка завантаження токенізатора: {e}")
    raise SystemExit(1)

# ---------------------------
# Очистка старых датасетов
# ---------------------------
for folder in os.listdir(pyIanthe_config.FOLDER_CORPUS):
    full_path = os.path.join(pyIanthe_config.FOLDER_CORPUS, folder)
    if "_TO_DELETE_" in folder and os.path.isdir(full_path):
        try:
            shutil.rmtree(full_path, ignore_errors=True)
            print(f"[INFO] Видалено старий датасет: {full_path}")
        except Exception as e:
            print(f"[WARNING] Не вдалося видалити старий датасет: {full_path}, причина: {e}")

print("\n[INFO] Скрипт виконано успішно.")
