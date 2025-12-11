# pyIanthe_train.py
import os
import json
import shutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config
)
from datasets import load_from_disk
import pyIanthe_config

# ============================================================
# Основной блок запуска (для Windows multiprocessing)
# ============================================================
if __name__ == "__main__":

    # ============================================================
    # 0. Перевірка GPU та налаштування
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[INFO] GPU знайдено:", torch.cuda.get_device_name(0))
        fp16 = pyIanthe_config.FP16
        pin_memory = pyIanthe_config.PIN_MEMORY
    else:
        print("[INFO] GPU не знайдено, тренування буде на CPU")
        fp16 = False
        pin_memory = False

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[INFO] Кількість доступних GPU: {num_gpus}")

    # ============================================================
    # 1. Чекпоінти (resume / нове тренування)
    # ============================================================
    CHECKPOINT_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_CHECKPOINTS)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def list_checkpoint_dirs():
        if not os.path.isdir(CHECKPOINT_DIR):
            return []
        items = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
        def keyfn(x):
            parts = x.split("-")
            try:
                return int(parts[-1])
            except:
                return x
        return sorted(items, key=keyfn)

    def get_last_checkpoint():
        lst = list_checkpoint_dirs()
        if not lst:
            return None
        return os.path.join(CHECKPOINT_DIR, lst[-1])

    def prepare_checkpoints_menu(auto_resume=True):
        last = get_last_checkpoint()
        if last is None:
            print("[INFO] Чекпоінтів не знайдено — буде створено нове тренування.")
            return None

        if auto_resume:
            print(f"[INFO] Знайдено останній чекпоінт: {last}, тренування продовжиться автоматично.")
            return last

        # --- меню для ручного вибору
        while True:
            print("\n=== Меню керування чекпоінтами ===")
            print(f"Знайдено останній чекпоінт: {last}")
            print("1) Продовжити тренування з останнього чекпоінта")
            print("2) Почати тренування заново (видалити старі чекпоінти)")
            print("3) Видалити всі чекпоінти і вийти")
            print("4) Вийти, нічого не змінюючи")
            choice = input("Оберіть дію (1-4): ").strip()
            if choice == "1":
                return last
            if choice == "2":
                shutil.rmtree(CHECKPOINT_DIR)
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                return None
            if choice == "3":
                shutil.rmtree(CHECKPOINT_DIR)
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                exit(0)
            if choice == "4":
                exit(0)
            print("[WARN] Невірний вибір. Спробуйте ще раз.")

    # Використовуємо авто-резюме
    resume_checkpoint = prepare_checkpoints_menu(auto_resume=True)

    # ============================================================
    # 2. Конфігурація навчання та директорії
    # ============================================================
    EPOCHS = pyIanthe_config.EPOCHS
    LEARNING_RATE = pyIanthe_config.LEARNING_RATE
    PER_DEVICE_BATCH_SIZE = pyIanthe_config.PER_DEVICE_BATCH_SIZE
    CONTEXT_LENGTH = pyIanthe_config.CONTEXT_LENGTH
    SAVE_STEPS = pyIanthe_config.SAVE_STEPS

    TRAINING_CONFIG = {
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.9,
        "num_train_epochs": EPOCHS
    }

    REPORTS_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_REPORTS)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ============================================================
    # 3. Завантаження датасету та токенізатора
    # ============================================================
    train_dataset_path = os.path.join(pyIanthe_config.FOLDER_TRAIN_DATASET, pyIanthe_config.TRAIN_DATASET_FILENAME)
    dataset = load_from_disk(train_dataset_path)
    print(f"[INFO] Завантажено датасет з {len(dataset)} записами")

    model_parts = pyIanthe_config.MODEL_ID.split("/")
    tokenizer_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, *model_parts)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        tokenizer.model_max_length = CONTEXT_LENGTH
        print("[INFO] Токенізатор завантажено успішно.")
    except Exception as e:
        print("[ERROR] Помилка завантаження токенізатора:", e)
        exit(1)

    # Фільтрація некоректних текстів
    def is_valid_text(example):
        text = example.get("text", "")
        return isinstance(text, str) and len(text.strip()) > 0

    dataset = dataset.filter(is_valid_text)
    print(f"[INFO] Після фільтрації датасет містить {len(dataset)} записів")

    # Токенізація датасету
    def tokenize_function(examples):
        try:
            return tokenizer(examples["text"], truncation=True, max_length=CONTEXT_LENGTH)
        except Exception as e:
            print("[WARN] Помилка токенізації:", e)
            return {"input_ids": [], "attention_mask": []}

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ============================================================
    # 4. Створення моделі
    # ============================================================
    vocab_size = len(tokenizer)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=pyIanthe_config.EMBEDDING_DIM,
        n_layer=pyIanthe_config.NUM_LAYERS,
        n_head=pyIanthe_config.HEADS,
    )
    model = AutoModelForCausalLM.from_config(config).to(device)

    # ============================================================
    # 5. Налаштування тренування
    # ============================================================
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=pyIanthe_config.SAVE_LIMIT,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY,
        fp16=fp16,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=pyIanthe_config.NUM_WORKERS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # ============================================================
    # 6. Генерація звітів і метрики
    # ============================================================
    GENERATE_EXAMPLES = ["Hello, how are you?", "Once upon a time,", "The quick brown fox"]

    def compute_perplexity(logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return torch.exp(loss.mean()).item()

    def compute_metrics(text):
        tokens = text.split()
        total_tokens = len(tokens)
        meaningful_tokens = sum(1 for t in tokens if t.isalnum())
        perc_meaningful = meaningful_tokens / total_tokens * 100 if total_tokens else 0
        return total_tokens, perc_meaningful

    def generate_report(epoch_index: int):
        model.eval()
        report = {}
        for prompt in GENERATE_EXAMPLES:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=TRAINING_CONFIG["max_new_tokens"],
                    temperature=TRAINING_CONFIG["temperature"],
                    top_p=TRAINING_CONFIG["top_p"],
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                avg_len, perc_meaningful = compute_metrics(text)

                logits = model(**inputs).logits
                ppl = compute_perplexity(logits, inputs["input_ids"])

                report[prompt] = {
                    "text": text,
                    "perplexity": ppl,
                    "avg_length": avg_len,
                    "perc_meaningful_tokens": perc_meaningful
                }

        report_file = os.path.join(REPORTS_DIR, f"report_epoch_{epoch_index+1}.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Звіт по епосі {epoch_index+1} збережено: {report_file}")

    # ============================================================
    # 7. Запуск тренування
    # ============================================================
    print(f"[INFO] Старт тренування. resume_from_checkpoint = {resume_checkpoint}")
    first_resume = resume_checkpoint

    for epoch in range(EPOCHS):
        print(f"===== Епоха {epoch+1} / {EPOCHS} =====")
        trainer.train(resume_from_checkpoint=first_resume)
        first_resume = None  # далі trainer продовжує з поточного стану

        # Збереження окремої епохи
        epoch_dir = os.path.join(CHECKPOINT_DIR, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        trainer.save_model(epoch_dir)

        # Стан тренера
        try:
            trainer.save_state()
        except Exception as e:
            print(f"[WARN] Не вдалося викликати trainer.save_state(): {e}")

        # Переміщуємо можливі файли стану
        for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            src = os.path.join(CHECKPOINT_DIR, fname)
            dst = os.path.join(epoch_dir, fname)
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Генеруємо звіт
        generate_report(epoch)
        print(f"[INFO] Epoch {epoch+1} збережено у: {epoch_dir}")

    print("[INFO] Тренування завершено. Всі звіти та чекпоінти збережені.")
