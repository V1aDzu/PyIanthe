import os
import re
import json
import time
import shutil
from transformers import AutoTokenizer
import pyIanthe_config
# ---------------------------
# Настройки окружения HF
# ---------------------------
os.environ["HF_HOME"] = pyIanthe_config.HF_HOME
os.environ["HF_DATASETS_CACHE"] = pyIanthe_config.HF_DATASETS_CACHE
os.environ["HF_METRICS_CACHE"] = pyIanthe_config.HF_METRICS_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = pyIanthe_config.HF_HUB_WARN_DISABLE

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder

# ---------------------------
# Каталоги
# ---------------------------
os.makedirs(pyIanthe_config.FOLDER_CORPUS, exist_ok=True)
os.makedirs(pyIanthe_config.FOLDER_TRAIN_DATASET, exist_ok=True)
os.makedirs(getattr(pyIanthe_config, "FOLDER_EVAL_DATASET", "dataset/eval"), exist_ok=True)

SEED = 42  # фиксированный seed для shuffle

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
    local_records = 0
    if os.path.exists(local_path):
        try:
            ds = load_from_disk(local_path)
            local_records = len(ds)
            print(f"[INFO] Локальна копія знайдена: {local_path}, записів: {local_records}")
        except Exception as e:
            print(f"[WARNING] Не вдалося завантажити локальну копію: {e}")
            ds = None
            local_records = 0

    # --- Проверка доступного количества на HF ---
    if "hf_id" in info and info["hf_id"]:
        try:
            builder = load_dataset_builder(info["hf_id"])
            if split_name in builder.info.splits:
                hf_total = builder.info.splits[split_name].num_examples
                print(f"[INFO] Доступно записів у спліті '{split_name}': {hf_total}")
        except Exception:
            print(f"[WARNING] Не вдалося визначити максимальну кількість записів на HF")
            hf_total = None

    # --- Выбор действия ---
    if ds is None:
        if not ("hf_id" in info and info["hf_id"]):
            print(f"[INFO] Для {dataset_name} не задан hf_id, пропускаємо завантаження")
            continue
        action = input(f"[PROMPT] Доступні опції: (d) Завантажити, (k) Пропустити: ").strip().lower()
        if action != "d":
            print(f"[INFO] Пропускаємо {dataset_name}")
            continue
        action = "d"
    else:
        options = "(d) Завантажити/замінити, (k) Пропустити, (r) Видалити локальну копію"
        action = input(f"[PROMPT] Доступні опції: {options}: ").strip().lower()

    if action == "k":
        print(f"[INFO] Пропускаємо {dataset_name}")
        continue
    elif action in ("r", "d"):
        # --- Переименование старого датасета вместо удаления ---
        if os.path.exists(local_path):
            try:
                timestamp = int(time.time())
                backup_path = f"{local_path}_TO_DELETE_{timestamp}"
                os.rename(local_path, backup_path)
                print(f"[INFO] Старий датасет перейменовано для видалення: {backup_path}")
            except Exception as e:
                print(f"[WARNING] Не вдалося перейменувати старий датасет: {e}")

    if action == "d":
        # --- Проверка hf_id ---
        if not ("hf_id" in info and info["hf_id"]):
            print(f"[INFO] Для {dataset_name} не задан hf_id, пропускаємо завантаження")
            continue

        # --- Ввод количества записей ---
        n_str = input(f"Введіть бажану кількість записів: ").strip()
        try:
            n = int(n_str)
            if n <= 0:
                print("[INFO] Пропускаємо завантаження через некоректне число")
                continue
            if hf_total is not None and n > hf_total:
                n = hf_total
                print(f"[WARNING] Вказано більше, ніж доступно. Завантажено максимум: {n}")
        except ValueError:
            print("[INFO] Пропускаємо завантаження через некоректне число")
            continue

        # --- Скачивание нового датасета ---
        try:
            split_arg = f"{split_name}[:{n}]"
            ds = load_dataset(info["hf_id"], split=split_arg)
            ds.save_to_disk(local_path)
            print(f"[INFO] Датасет {dataset_name} завантажено локально, записів: {len(ds)}")
        except Exception as e:
            print(f"[ERROR] Не вдалося скачати {dataset_name}: {e}")
            ds = None

    if ds is not None:
        loaded_datasets.append(ds)
        print(f"[INFO] Додано до списку для об'єднання: {dataset_name}, записів: {len(ds)}")

# ---------------------------
# Объединение всех датасетов
# ---------------------------
if not loaded_datasets:
    print("[WARNING] Жоден датасет не було завантажено або знайдено локально. Завершення роботи.")
    raise SystemExit(1)

print("\n[INFO] Об'єднання всіх завантажених датасетів...")
combined_dataset = concatenate_datasets(loaded_datasets)
total_combined = len(combined_dataset)
print(f"[INFO] Загальний розмір об'єднаного датасету: {total_combined} записів")

# ---------------------------
# Перемешивание и деление на train/eval
# ---------------------------
eval_percent = getattr(pyIanthe_config, "EVAL_PERCENT", 5)
eval_fraction = float(eval_percent) / 100.0
eval_size = int(total_combined * eval_fraction)
if eval_size < 1: eval_size = 1
print(f"[INFO] Формуємо eval-набір: {eval_percent}% від {total_combined} = {eval_size} записів")

shuffled = combined_dataset.shuffle(seed=SEED)
eval_dataset = shuffled.select(range(eval_size))
train_dataset = shuffled.select(range(eval_size, total_combined))

train_dataset.save_to_disk(pyIanthe_config.FOLDER_TRAIN_DATASET)
print(f"[INFO] Train збережено: {pyIanthe_config.FOLDER_TRAIN_DATASET}, записів: {len(train_dataset)}")
eval_dataset.save_to_disk(pyIanthe_config.FOLDER_EVAL_DATASET)
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
    sample_texts = [
        str(train_dataset[i]["text"])
        for i in range(sample_count)
        if train_dataset[i] is not None and "text" in train_dataset[i] and train_dataset[i]["text"]
    ]

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
# Очистка старых датасетов после завершения
# ---------------------------
for folder in os.listdir(pyIanthe_config.FOLDER_CORPUS):
    full_path = os.path.join(pyIanthe_config.FOLDER_CORPUS, folder)
    if "_TO_DELETE_" in folder and os.path.isdir(full_path):
        try:
            shutil.rmtree(full_path)
            print(f"[INFO] Видалено старий датасет: {full_path}")
        except Exception as e:
            print(f"[WARNING] Не вдалося видалити старий датасет: {full_path}, причина: {e}")

print("\n[INFO] Скрипт виконано успішно.")
