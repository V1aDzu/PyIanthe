import os
import re
import json
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    load_dataset_builder
)
from transformers import AutoTokenizer
import pyIanthe_config

# ---------------------------
# Налаштування / каталоги
# ---------------------------
os.makedirs(pyIanthe_config.FOLDER_CORPUS, exist_ok=True)
os.makedirs(pyIanthe_config.FOLDER_TRAIN_DATASET, exist_ok=True)
os.makedirs(getattr(pyIanthe_config, "FOLDER_EVAL_DATASET", "dataset/eval"), exist_ok=True)

SEED = 42  # фіксований seed для повторюваності

# ---------------------------
# Утиліта: витягнути ім'я спліту
# ---------------------------
def extract_split_name(split_string):
    """
    'train[:50000]' -> 'train'
    'train[10%:20%]' -> 'train'
    'validation[:100]' -> 'validation'
    'test' -> 'test'
    """
    if not isinstance(split_string, str):
        return "train"
    match = re.match(r"^([a-zA-Z_]+)", split_string)
    if match:
        return match.group(1)
    return "train"

# ---------------------------
# Зчитати метадані про корпуси
# ---------------------------
corpus_info_path = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.CORPUS_DATA_FILENAME)
with open(corpus_info_path, "r", encoding="utf-8") as f:
    dataset_info = json.load(f)

loaded_datasets = []

# ---------------------------
# Проходимо по всім записам конфігу
# ---------------------------
for info in dataset_info:
    print(f"\n[INFO] Обробка датасету: {info.get('name', 'UNKNOWN')}")
    local_path = os.path.join(pyIanthe_config.FOLDER_CORPUS, info.get("name", "unknown"))

    # Форматування поліглоду
    langs = info.get("lang", [])
    lang_str = ", ".join(langs) if isinstance(langs, list) else langs
    clean = info.get("clean", "unknown")
    description = info.get("info", "")
    purpose = info.get("purpose", "unknown")

    print(f"[INFO] Доступні мови: {lang_str}")
    print(f"[INFO] Якість: {clean}")
    print(f"[INFO] Опис/особливості: {description}")
    print(f"[INFO] Призначення: {purpose}")

    ds = None
    total_records = "unknown"
    avg_length = 0
    approx_size_mb = "unknown"

    # --- Якщо локальна копія існує — завантажити її швидко ---
    if os.path.exists(local_path):
        try:
            ds = load_from_disk(local_path)
            total_records = len(ds)
            sample_count = min(100, total_records)
            if sample_count > 0:
                avg_length = sum(len(ds[i].get("text", "")) for i in range(sample_count)) / sample_count
                approx_size_mb = total_records * avg_length / (1024 * 1024)
                approx_size_mb = f"{approx_size_mb:.2f} MB"
            else:
                avg_length = 0
                approx_size_mb = "0.00 MB"
            print(f"[INFO] Локальна копія знайдена: {local_path}, розмір: {total_records} записів, приблизний обсяг: {approx_size_mb}, середня довжина тексту: {avg_length:.1f} символів")
        except Exception as e:
            print(f"[WARNING] Не вдалося оцінити локальну копію: {e}")
            ds = None
    else:
        # --- Швидка та безпечна оцінка через builder (без streaming) ---
        if "hf_id" in info:
            try:
                builder = load_dataset_builder(info["hf_id"])
                raw_split = info.get("split", "train")
                split_name = extract_split_name(raw_split)

                if split_name in builder.info.splits:
                    total_records = builder.info.splits[split_name].num_examples
                    print(f"[INFO] Швидка оцінка: ~{total_records} записів у спліті '{split_name}'")
                else:
                    total_records = "unknown"
                    print(f"[INFO] Швидка оцінка: unknown — спліт '{split_name}' відсутній у датасеті")

                avg_length = 0
                approx_size_mb = "unknown"

            except Exception as e:
                print(f"[WARNING] Не вдалося отримати метадані через load_dataset_builder: {e}")
                total_records = "unknown"
                avg_length = 0
                approx_size_mb = "unknown"

            # --- Питати користувача чи завантажувати ---
            answer = input(f"[PROMPT] Датасету {info.get('name')} немає на диску. Завантажити? (y/n): ").strip().lower()
            if answer == "y":
                # Вибір способу завантаження: процент або кількість записів
                choice = input("Ввести 'p' для відсотку або 'n' для кількості записів: ").strip().lower()
                split_arg = info.get("split", "train[:1%]")

                if choice == "p":
                    perc = float(input("Введіть відсоток датасету для завантаження: ").strip())
                    split_arg = f"train[:{perc}%]"
                elif choice == "n":
                    if total_records != "unknown":
                        print(f"[INFO] Максимальна кількість записів, яку можна взяти: {total_records}")
                    n = int(input("Введіть кількість записів для завантаження: ").strip())
                    if total_records != "unknown" and n > total_records:
                        print(f"[WARNING] Вказана кількість більша за доступну, буде використано максимум {total_records}")
                        n = total_records
                    split_arg = f"train[:{n}]"
                else:
                    # Якщо ввели щось інше — використовуємо дефолтний split з конфига
                    split_arg = info.get("split", "train[:1%]")

                # --- Завантаження та збереження ---
                try:
                    print(f"[INFO] Починаю завантаження {info.get('hf_id')} зі сплітом '{split_arg}' ...")
                    ds = load_dataset(info["hf_id"], split=split_arg)
                    ds.save_to_disk(local_path)
                    print(f"[INFO] Датасет {info.get('name')} збережено локально у {local_path}, записів: {len(ds)}")
                except Exception as e:
                    print(f"[ERROR] Не вдалося завантажити {info.get('name')}: {e}")
                    ds = None
            else:
                print(f"[INFO] Пропускаємо датасет {info.get('name')}")
        else:
            print(f"[WARNING] Датасет {info.get('name')} недоступний для завантаження (немає hf_id). Пропускаємо.")

    # --- Якщо датасет вдалося отримати (локально або завантажено) — додаємо до списку ---
    if ds is not None:
        loaded_datasets.append(ds)
        print(f"[INFO] Додано до списку для об'єднання: {info.get('name')}, розмір: {len(ds)} записів")

# ---------------------------
# Перевірка: чи є що об'єднувати
# ---------------------------
if not loaded_datasets:
    print("[WARNING] Жоден датасет не було завантажено або знайдено локально. Завершення роботи.")
    raise SystemExit(1)

# ---------------------------
# Об'єднати всі датасети
# ---------------------------
print("\n[INFO] Об'єднання всіх завантажених датасетів...")
combined_dataset = concatenate_datasets(loaded_datasets)
total_combined = len(combined_dataset)
print(f"[INFO] Загальний розмір об'єднаного датасету: {total_combined} записів")

# ---------------------------
# Перемішати та розрізати на train / eval (без перетину)
# ---------------------------
eval_percent = getattr(pyIanthe_config, "EVAL_PERCENT", 5)
eval_fraction = float(eval_percent) / 100.0
eval_size = int(total_combined * eval_fraction)

if eval_size < 1:
    eval_size = 1

print(f"[INFO] Формуємо eval-набір: {eval_percent}% від {total_combined} = {eval_size} записів")

# Перемішуємо для випадкового пропорційного вибору (фіксований seed для повторюваності)
shuffled = combined_dataset.shuffle(seed=SEED)

# Витягуємо eval та train без перекриття
eval_dataset = shuffled.select(range(eval_size))
train_dataset = shuffled.select(range(eval_size, total_combined))

# ---------------------------
# Зберегти train та eval
# ---------------------------
train_dataset_path = pyIanthe_config.FOLDER_TRAIN_DATASET
eval_dataset_path = pyIanthe_config.FOLDER_EVAL_DATASET

print(f"[INFO] Збереження train у: {train_dataset_path} (з урахуванням видалення eval)")
train_dataset.save_to_disk(train_dataset_path)
print(f"[INFO] Train збережено: {train_dataset_path}, записів: {len(train_dataset)}")

print(f"[INFO] Збереження eval у: {eval_dataset_path}")
eval_dataset.save_to_disk(eval_dataset_path)
print(f"[INFO] Eval збережено: {eval_dataset_path}, записів: {len(eval_dataset)}")

# ---------------------------
# Перевірка токенізатора на train-наборі
# ---------------------------
tokenizer_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, "Qwen", "Qwen2.5-0.5B-Instruct")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    print("[INFO] Токенізатор завантажено успішно.")
    print(f"[INFO] Розмір словника: {tokenizer.vocab_size}")

    # Тестова токенізація декількох прикладів з train (або з комбінованого)
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

print("\n[INFO] Скрипт виконано успішно.")
