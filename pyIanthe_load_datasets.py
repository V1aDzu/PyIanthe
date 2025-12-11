import os
import json
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import pyIanthe_config

# --- Каталоги ---
os.makedirs(pyIanthe_config.FOLDER_CORPUS, exist_ok=True)
os.makedirs(pyIanthe_config.FOLDER_TRAIN_DATASET, exist_ok=True)

# --- Загрузка информации о датасетах ---
corpus_info_path = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.CORPUS_DATA_FILENAME)
with open(corpus_info_path, "r", encoding="utf-8") as f:
    dataset_info = json.load(f)

all_datasets = []

for info in dataset_info:
    print(f"\n[INFO] Обробка датасету: {info['name']}")
    local_path = os.path.join(pyIanthe_config.FOLDER_CORPUS, info['name'])

    # --- Форматирование информации ---
    langs = info.get("lang", [])
    lang_str = ", ".join(langs) if isinstance(langs, list) else langs
    clean = info.get("clean", "unknown")
    description = info.get("info", "")
    purpose = info.get("purpose", "unknown")

    print(f"[INFO] Доступні мови: {lang_str}")
    print(f"[INFO] Якість: {clean}")
    print(f"[INFO] Описание/особливості: {description}")
    print(f"[INFO] Призначення: {purpose}")

    ds = None
    total_records = "unknown"
    avg_length = 0
    approx_size_mb = "unknown"

    # --- Локальная копия ---
    if os.path.exists(local_path):
        try:
            ds = load_from_disk(local_path)
            total_records = len(ds)
            sample_count = min(100, total_records)
            avg_length = sum(len(ds[i].get("text", "")) for i in range(sample_count)) / sample_count
            approx_size_mb = total_records * avg_length / (1024 * 1024)
            approx_size_mb = f"{approx_size_mb:.2f} MB"
            print(f"[INFO] Локальна копія знайдена: {local_path}, розмір: {total_records} записів, приблизний обсяг: {approx_size_mb}, середня довжина тексту: {avg_length:.1f} символів")
        except Exception as e:
            print(f"[WARNING] Не вдалося оцінити розмір локальної копії: {e}")
    else:
        # --- Оценка размера через streaming ---
        if "hf_id" in info:
            try:
                print("[INFO] Оцінка кількості записів та середньої довжини тексту (streaming, перші 100 записів)...")
                ds_stream = load_dataset(info["hf_id"], split="train", streaming=True)
                total_records = 0
                sample_lengths = []
                n_sample = 100
                for i, item in enumerate(ds_stream):
                    total_records += 1
                    sample_lengths.append(len(item.get("text", "")))
                    if i >= n_sample - 1:
                        break
                avg_length = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
                approx_size_mb = total_records * avg_length / (1024 * 1024)
                approx_size_mb = f"{approx_size_mb:.2f} MB"
                print(f"[INFO] Приблизний розмір: {approx_size_mb}, кількість записів: {total_records}, середня довжина тексту: {avg_length:.1f} символів")
            except Exception as e:
                print(f"[WARNING] Не вдалося оцінити розмір: {e}")
                total_records = "unknown"
                avg_length = 0
                approx_size_mb = "unknown"

            answer = input(f"[PROMPT] Датасету {info['name']} немає на диску. Завантажити? (y/n): ").strip().lower()
            if answer == "y":
                # --- Ввод пользователем сколько записей или процент ---
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

                # --- Загрузка и сохранение ---
                try:
                    ds = load_dataset(info["hf_id"], split=split_arg)
                    ds.save_to_disk(local_path)
                    print(f"[INFO] Датасет {info['name']} збережено локально у {local_path}, записів: {len(ds)}")
                except Exception as e:
                    print(f"[ERROR] Не вдалося завантажити {info['name']}: {e}")
                    ds = None
            else:
                print(f"[INFO] Пропускаємо датасет {info['name']}")
        else:
            print(f"[WARNING] Датасет {info['name']} недоступний для завантаження (немає hf_id). Пропускаємо.")
    
    if ds is not None:
        all_datasets.append(ds)
        print(f"[INFO] Датасет {info['name']} додано для об’єднання, розмір: {len(ds)} записів")

# --- Объединение всех загруженных датасетов ---
if all_datasets:
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"\n[INFO] Загальний розмір об’єднаного датасету: {len(combined_dataset)} записів")
    train_dataset_path = os.path.join(pyIanthe_config.FOLDER_TRAIN_DATASET, pyIanthe_config.TRAIN_DATASET_FILENAME)
    combined_dataset.save_to_disk(train_dataset_path)
    print(f"[INFO] Об’єднаний датасет збережено: {train_dataset_path}")
else:
    print("[WARNING] Жоден датасет не було завантажено. Скрипт завершив роботу.")
    exit(1)

# --- Перевірка токенізатора ---
tokenizer_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, "Qwen", "Qwen2.5-0.5B-Instruct")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    print("[INFO] Токенізатор завантажено успішно.")
    print("[INFO] Розмір словника:", tokenizer.vocab_size)

    # --- Перевірка токенізації декількох записів ---
    sample_texts = [combined_dataset[i]["text"] for i in range(min(5, len(combined_dataset)))]
    tokenized_samples = tokenizer(sample_texts, truncation=True, padding=True)
    print("[INFO] Токенізація тестових записів пройшла успішно.")
    for i, t in enumerate(sample_texts):
        print(f"\n[Sample {i+1}] оригінал: {t[:100]}...")
        print(f"[Sample {i+1}] токени: {tokenized_samples['input_ids'][i][:10]} ...")
except Exception as e:
    print("[ERROR] Помилка завантаження токенізатора:", e)
    exit(1)
