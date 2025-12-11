# pyIanthe_train_final.py (CORRECT LOGIC)
import os
import sys
import json
import torch
import shutil
import warnings
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

# ==================== НАЛАШТУВАННЯ WARNINGS ====================
if not pyIanthe_config.DEBUG:
    import warnings
    warnings.filterwarnings("ignore")
    
    import logging
    # Вимкнути ВСІ warnings від transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)
    
    # Встановити глобальну verbosity transformers на мінімум
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    
    print("[INFO] DEBUG режим вимкнено, warnings приховані")
else:
    print("[INFO] DEBUG режим увімкнено, всі warnings показуються")
# ================================================================

if __name__ == "__main__":

    # 0. Перевірка GPU та налаштування
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = pyIanthe_config.FP16 if device == "cuda" else False
    pin_memory = pyIanthe_config.PIN_MEMORY if device == "cuda" else False
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[INFO] Пристрій: {device}, GPU: {num_gpus}")
    if device == "cuda":
        print("[INFO] GPU знайдено:", torch.cuda.get_device_name(0))
    else:
        print("[INFO] Тренування буде на CPU")

    # 1. Папки
    CHECKPOINT_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_CHECKPOINTS)
    MAIN_MODEL_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_MODEL)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MAIN_MODEL_DIR, exist_ok=True)

    # Функція для знаходження останнього чекпоінта
    def get_last_checkpoint():
        if not os.path.isdir(CHECKPOINT_DIR):
            return None
        dirs = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
        if not dirs:
            return None
        try:
            # Шукаємо checkpoint-1, checkpoint-2 тощо
            checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                dirs_sorted = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                return os.path.join(CHECKPOINT_DIR, dirs_sorted[-1])
        except:
            pass
        # Якщо не знайшли checkpoint-X, беремо останній за алфавітом
        dirs_sorted = sorted(dirs)
        return os.path.join(CHECKPOINT_DIR, dirs_sorted[-1])

    last_checkpoint = get_last_checkpoint()

    # 2. Конфігурація навчання
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

    # 3. Завантаження датасету та токенізатора (з базової моделі в models/)

    train_dataset_path = pyIanthe_config.FOLDER_TRAIN_DATASET

    # Перевірка наявності датасету
    if not os.path.exists(train_dataset_path):
        print(f"[ERROR] Папка датасету не знайдена: {train_dataset_path}")
        print(f"[INFO] Спочатку запустіть pyIanthe_load_dataset.py для завантаження датасетів")
        sys.exit(1)

    # Перевірка чи є arrow файли (datasets зберігає дані в arrow форматі)
    arrow_files = [f for f in os.listdir(train_dataset_path) if f.endswith('.arrow')]
    dataset_info = os.path.join(train_dataset_path, 'dataset_info.json')

    if not arrow_files and not os.path.exists(dataset_info):
        print(f"[ERROR] Папка {train_dataset_path} порожня або не містить датасет")
        print(f"[INFO] Знайдено файлів: {os.listdir(train_dataset_path) if os.path.exists(train_dataset_path) else 'папка не існує'}")
        print(f"[INFO] Спочатку запустіть pyIanthe_load_dataset.py для завантаження датасетів")
        sys.exit(1)

    dataset = load_from_disk(train_dataset_path)
    print(f"[INFO] Датасет завантажено, записів: {len(dataset)}")

    # ТОКЕНІЗАТОР БЕРЕТЬСЯ З БАЗОВОЇ МОДЕЛІ (models/Qwen/...)
    tokenizer_source_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, *pyIanthe_config.MODEL_ID.split("/"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_dir, local_files_only=True)
    tokenizer.model_max_length = CONTEXT_LENGTH
    print(f"[INFO] Токенізатор завантажено з: {tokenizer_source_dir}")

    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    print(f"[INFO] Після фільтрації: {len(dataset)} записів")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=CONTEXT_LENGTH)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. ЛОГІКА ЗАВАНТАЖЕННЯ МОДЕЛІ:
    # Пріоритет: checkpoint/ → model/ → нова модель
    vocab_size = len(tokenizer)
    
    # Перевіряємо наявність файлів моделі
    checkpoint_has_model = last_checkpoint and os.path.exists(os.path.join(last_checkpoint, "model.safetensors"))
    main_model_exists = os.path.exists(os.path.join(MAIN_MODEL_DIR, "model.safetensors"))
    
    if checkpoint_has_model:
        # 1. Є чекпоінт → завантажуємо з нього
        print(f"[INFO] Знайдено чекпоінт з моделлю: {last_checkpoint}")
        print(f"[INFO] Завантажуємо модель з чекпоінта (навчання продовжиться)")
        # ВИПРАВЛЕННЯ: додаємо ignore_mismatched_sizes для tie_word_embeddings
        model = AutoModelForCausalLM.from_pretrained(
            last_checkpoint, 
            ignore_mismatched_sizes=False,
            local_files_only=True
        ).to(device)
        
        # Перевіряємо чи lm_head має правильні ваги (через tie_word_embeddings)
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            print("[INFO] ✓ Прив'язані ваги lm_head оновлено")
        
        resume_from = last_checkpoint
        
    elif main_model_exists:
        # 2. Немає чекпоінта, але є model/ → завантажуємо звідти
        print(f"[INFO] Чекпоінта немає, але знайдено модель у: {MAIN_MODEL_DIR}")
        print(f"[INFO] Завантажуємо модель (optimizer почнеться з нуля)")
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL_DIR,
            ignore_mismatched_sizes=False,
            local_files_only=True
        ).to(device)
        
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            print("[INFO] ✓ Прив'язані ваги lm_head оновлено")
        
        resume_from = None  # Optimizer/scheduler не відновлюються
        
    else:
        # 3. Нічого немає → створюємо нову модель
        print("[INFO] Не знайдено ні чекпоінта, ні моделі")
        print("[INFO] Створюємо нову модель з конфігурації GPT2")
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=CONTEXT_LENGTH,
            n_ctx=CONTEXT_LENGTH,
            n_embd=pyIanthe_config.EMBEDDING_DIM,
            n_layer=pyIanthe_config.NUM_LAYERS,
            n_head=pyIanthe_config.HEADS,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
            # Виправлення попередження про loss_type
            loss_type=None,  # Використовувати ForCausalLMLoss (default)
        )
        model = AutoModelForCausalLM.from_config(config).to(device)
        resume_from = None

    print(f"[INFO] Модель завантажена, параметрів: {model.num_parameters():,}")
    
    # Перевіряємо чи lm_head прив'язаний до embeddings
    if hasattr(model, 'lm_head') and hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'wte'):
            lm_head_ptr = model.lm_head.weight.data_ptr()
            wte_ptr = model.transformer.wte.weight.data_ptr()
            if lm_head_ptr == wte_ptr:
                print("[INFO] ✓ lm_head.weight правильно прив'язаний до wte (tie_word_embeddings працює)")
            else:
                print("[WARN] ⚠ lm_head.weight НЕ прив'язаний до wte (можлива проблема)")


    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=1,  # Тренуємо по 1 епосі за раз
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=pyIanthe_config.SAVE_LIMIT,
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY,
        fp16=fp16,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 6. Метрики та звіти
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
        print(f"[INFO] Звіт по епосі {epoch_index+1} збережено: {report_file}")

    # 7. Тренування
    print(f"\n[INFO] Старт тренування на {EPOCHS} епох(и)")

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"===== Епоха {epoch+1} / {EPOCHS} =====")
        print(f"{'='*60}")

        try:
            # Тренуємо 1 епоху
            # resume_from використовується тільки для ПЕРШОЇ епохи якщо був чекпоінт
            if epoch == 0 and resume_from:
                print(f"[INFO] Продовжуємо з чекпоінта: {resume_from}")
                trainer.train(resume_from_checkpoint=resume_from)
            else:
                trainer.train()

        except KeyboardInterrupt:
            print(f"\n[WARN] ⚠ Тренування перервано користувачем (Ctrl+C)")
            print(f"[INFO] Зберігаємо поточний стан...")
            
            # Зберігаємо головну модель
            model.save_pretrained(MAIN_MODEL_DIR)
            tokenizer.save_pretrained(MAIN_MODEL_DIR)
            print(f"[INFO] ✓ Модель збережена у: {MAIN_MODEL_DIR}")
            
            # Зберігаємо аварійний чекпоінт
            emergency_checkpoint = os.path.join(CHECKPOINT_DIR, f"checkpoint-interrupted-epoch{epoch+1}")
            os.makedirs(emergency_checkpoint, exist_ok=True)
            trainer.save_model(emergency_checkpoint)
            trainer.save_state()
            
            for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"]:
                src = os.path.join(CHECKPOINT_DIR, fname)
                dst = os.path.join(emergency_checkpoint, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            
            print(f"[INFO] ✓ Аварійний чекпоінт: {emergency_checkpoint}")
            print(f"[INFO] Для продовження запустіть скрипт знову")
            print("="*60)
            sys.exit(0)

        # Після кожної епохи зберігаємо:
        
        # 1) Головну модель у model/ (для використання/продовження без чекпоінта)
        print(f"[INFO] Зберігаємо головну модель у: {MAIN_MODEL_DIR}")
        
        # Переконуємось що config має правильний tie_word_embeddings
        if hasattr(model.config, 'tie_word_embeddings'):
            model.config.tie_word_embeddings = True
        
        model.save_pretrained(MAIN_MODEL_DIR)
        tokenizer.save_pretrained(MAIN_MODEL_DIR)
        
        # 2) Чекпоінт у checkpoints/checkpoint-X/ (для точного відновлення)
        checkpoint_epoch_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch+1}")
        print(f"[INFO] Зберігаємо чекпоінт у: {checkpoint_epoch_dir}")
        os.makedirs(checkpoint_epoch_dir, exist_ok=True)
        
        # Переконуємось що config має правильний tie_word_embeddings
        if hasattr(model.config, 'tie_word_embeddings'):
            model.config.tie_word_embeddings = True
        
        # Зберігаємо модель + config
        trainer.save_model(checkpoint_epoch_dir)
        
        # Зберігаємо стан trainer (optimizer, scheduler, rng)
        trainer.save_state()
        
        # Копіюємо файли стану у checkpoint
        for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"]:
            src = os.path.join(CHECKPOINT_DIR, fname)
            dst = os.path.join(checkpoint_epoch_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  ✓ Скопійовано {fname}")

        # Генеруємо звіт
        generate_report(epoch)
        
        print(f"\n[SUCCESS] Епоха {epoch+1} завершена і збережена")
        print(f"  → Модель: {MAIN_MODEL_DIR}")
        print(f"  → Чекпоінт: {checkpoint_epoch_dir}")

    print("\n" + "="*60)
    print("[SUCCESS] Тренування завершено!")
    print(f"Всього епох: {EPOCHS}")
    print(f"Фінальна модель: {MAIN_MODEL_DIR}")
    print(f"Звіти: {REPORTS_DIR}")
    print("="*60)
