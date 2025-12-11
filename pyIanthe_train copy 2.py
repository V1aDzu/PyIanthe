# pyIanthe_train_optimized.py
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

    # 0. Проверка GPU и настройки
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = pyIanthe_config.FP16 if device == "cuda" else False
    pin_memory = pyIanthe_config.PIN_MEMORY if device == "cuda" else False
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[INFO] Устройство: {device}, GPU: {num_gpus}")
    if device == "cuda":
        print("[INFO] GPU найдено:", torch.cuda.get_device_name(0))
    else:
        print("[INFO] Тренировка будет на CPU")

    # 1. Чекпоинты
    CHECKPOINT_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_CHECKPOINTS)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def list_checkpoint_dirs():
        if not os.path.isdir(CHECKPOINT_DIR):
            return []
        dirs = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
        try:
            return sorted(dirs, key=lambda x: int(x.split("-")[-1]))
        except:
            return sorted(dirs)

    def get_last_checkpoint():
        lst = list_checkpoint_dirs()
        if not lst:
            return None
        return os.path.join(CHECKPOINT_DIR, lst[-1])

    resume_checkpoint = get_last_checkpoint()
    if resume_checkpoint:
        print(f"[INFO] Найден последний чекпоинт: {resume_checkpoint}, обучение продолжится.")
    else:
        print("[INFO] Чекпоинтов нет, создается новое обучение.")
        resume_checkpoint = None

    # 2. Конфигурация обучения
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

    # 3. Загрузка датасета и токенизатора
    train_dataset_path = os.path.join(pyIanthe_config.FOLDER_TRAIN_DATASET, pyIanthe_config.TRAIN_DATASET_FILENAME)
    dataset = load_from_disk(train_dataset_path)
    print(f"[INFO] Датасет загружен, записей: {len(dataset)}")

    model_parts = pyIanthe_config.MODEL_ID.split("/")
    tokenizer_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, *model_parts)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.model_max_length = CONTEXT_LENGTH
    print("[INFO] Токенизатор загружен")

    # Фильтрация текстов
    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    print(f"[INFO] После фильтрации: {len(dataset)} записей")

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=CONTEXT_LENGTH)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Модель
    vocab_size = len(tokenizer)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=pyIanthe_config.EMBEDDING_DIM,
        n_layer=pyIanthe_config.NUM_LAYERS,
        n_head=pyIanthe_config.HEADS,
        use_cache=False,  # рекомендовано для тренування
        pad_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config).to(device)

    # 5. TrainingArguments с минимальными логами
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=pyIanthe_config.SAVE_LIMIT,
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY,
        fp16=fp16,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=0,  # безопасно для Windows
        report_to="none",          # отключаем лишние логеры HF
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 6. Метрики и отчеты
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

    def generate_report(epoch_index):
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
        print(f"[INFO] Отчет по эпохе {epoch_index+1} сохранен: {report_file}")

    # 7. Тренировка
    print(f"[INFO] Старт тренировки. resume_from_checkpoint = {resume_checkpoint}")
    first_resume = resume_checkpoint

    for epoch in range(EPOCHS):
        print(f"\n===== Эпоха {epoch+1} / {EPOCHS} =====")
        trainer.train(resume_from_checkpoint=first_resume)
        first_resume = None

        # Сохраняем модель отдельно для эпохи
        epoch_dir = os.path.join(CHECKPOINT_DIR, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        trainer.save_model(epoch_dir)

        # Сохраняем состояние тренера
        try:
            trainer.save_state()
        except Exception as e:
            print(f"[WARN] Не удалось вызвать trainer.save_state(): {e}")

        # Перемещаем файлы состояния
        for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            src = os.path.join(CHECKPOINT_DIR, fname)
            dst = os.path.join(epoch_dir, fname)
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Генерируем отчет
        generate_report(epoch)
        print(f"[INFO] Epoch {epoch+1} сохранена в: {epoch_dir}")

    print("[INFO] Тренировка завершена. Все отчеты и чекпоинты сохранены.")
