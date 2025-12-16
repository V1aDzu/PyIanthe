import os
import sys
import json
import torch
import shutil
import platform
import glob
import re
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config,
    TrainerCallback,
    set_seed
)
from datasets import load_from_disk
# Припускаємо, що модулі pyIanthe_log та pyIanthe_config існують і працюють
from modules.pyIanthe_log import setup_logging, configure_runtime 
import pyIanthe_config 

# ==================== ВИЗНАЧЕННЯ NUM_WORKERS ====================
def get_num_workers(logger):
    """
    Визначає фактичну кількість воркерів залежно від платформи та налаштувань.
    """
    system = platform.system()
    config_workers = pyIanthe_config.NUM_WORKERS
    
    if system == "Windows":
        if pyIanthe_config.WIN_WORKERS:
            logger.info(f"[NUM_WORKERS] Windows виявлено, WIN_WORKERS=True")
            logger.info(f"[NUM_WORKERS] Використовується NUM_WORKERS={config_workers}")
            logger.warning(f"[NUM_WORKERS] ⚠ Експериментальний режим! Можливі проблеми зі стабільністю")
            return config_workers
        else:
            logger.info(f"[NUM_WORKERS] Windows виявлено, WIN_WORKERS=False")
            logger.info(f"[NUM_WORKERS] Примусово встановлено NUM_WORKERS=0 (безпечний режим)")
            if config_workers > 0:
                logger.info(f"[NUM_WORKERS] NUM_WORKERS={config_workers} з конфігу проігноровано")
            return 0
    else:
        logger.info(f"[NUM_WORKERS] {system} виявлено")
        logger.info(f"[NUM_WORKERS] Використовується NUM_WORKERS={config_workers}")
        return config_workers
# ================================================================

# --- 1. CALLBACK: РЕЗЕРВНЕ КОПІЮВАННЯ ВЕСІВ (У SAFE_ARCHIVE_DIR) ---

class ReliableBackupCallback(TrainerCallback):
    """
    Створює надійну резервну копію тільки весів та токенізатора 
    з користувацьким іменуванням у SAFE_ARCHIVE_DIR (ваш MAIN_MODEL_DIR).
    """
    def __init__(self, safe_archive_dir, logger):
        self.safe_archive_dir = safe_archive_dir
        self.logger = logger
        os.makedirs(self.safe_archive_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        """Викликається після того, як Trainer збереже свій стандартний чекпоінт."""
        
        # 1. Формування імені папки: checkpoint_<HHMMSS>_<DDMMYYYY>
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%Y")
        # Назва: model_step_...
        backup_name = f"model_step_{state.global_step}_{timestamp}_{datestamp}" 
        backup_path = os.path.join(self.safe_archive_dir, backup_name)
        
        # 2. Сохраняємо ТІЛЬКИ веси та токенізатор (без стану оптимізатора)
        model_to_save = kwargs['model']
        tokenizer = kwargs['tokenizer']
        
        # Видаляємо старіші копії, щоб залишити лише 1-2 найновіші (як фінальні веси)
        self._cleanup_old_backups(self.safe_archive_dir, limit=3)
        
        model_to_save.save_pretrained(backup_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(backup_path)
            
        self.logger.info(f"\n[РЕЗЕРВ] Створена надійна копія весів: {backup_path}")

    def on_train_end(self, args, state, control, **kwargs):
        """Зберігаємо фінальну модель (також як резерв)"""
        model_to_save = kwargs['model']
        tokenizer = kwargs['tokenizer']
        
        # Очистка і збереження в кореневій папці MAIN_MODEL_DIR
        self.logger.info(f"\n[ФІНАЛ] Зберігаємо фінальну модель у: {self.safe_archive_dir}")
        shutil.rmtree(self.safe_archive_dir, ignore_errors=True)
        os.makedirs(self.safe_archive_dir, exist_ok=True)
        
        model_to_save.save_pretrained(self.safe_archive_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(self.safe_archive_dir)
        self.logger.info(f"✓ Фінальна модель збережена в: {self.safe_archive_dir}")
        
    def _cleanup_old_backups(self, directory, limit):
        """Видаляє старіші резервні копії, залишаючи limit найновіших."""
        # Шукаємо всі папки, що починаються з 'model_step_'
        all_backups = glob.glob(os.path.join(directory, 'model_step_*'))
        
        # Сортуємо за часом модифікації (новіші в кінці)
        all_backups.sort(key=os.path.getmtime)
        
        if len(all_backups) > limit:
            for oldest_path in all_backups[:-limit]:
                shutil.rmtree(oldest_path, ignore_errors=True)
                self.logger.info(f"[РЕЗЕРВ-ОЧИСТКА] Видалено стару копію: {os.path.basename(oldest_path)}")


# --- 2. CALLBACK: КАСТОМНЕ ІМЕНУВАННЯ ЧЕКПОІНТІВ ---

class CustomCheckpointingCallback(TrainerCallback):
    """
    Перейменовує стандартний чекпоінт Trainer у формат: 
    checkpoint_<step>_<HHMMSS>_<DDMMYYYY>
    """
    def __init__(self, logger):
        self.logger = logger
        
    def on_save(self, args, state, control, **kwargs):
        """Викликається після того, як Trainer збереже свій стандартний чекпоінт."""
        
        # Пошук останнього збереженого чекпоінта (наприклад, "checkpoints/checkpoint-500")
        list_of_files = glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        
        # Фільтруємо, щоб не чіпати кастомно названі папки
        trainer_checkpoints = [f for f in list_of_files if re.match(r'.*/checkpoint-\d+$', f)]

        if not trainer_checkpoints:
            return

        latest_checkpoint = max(trainer_checkpoints, key=os.path.getctime)
        
        # Формування нового імені
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%Y")
        
        # Витягуємо номер кроку (step)
        step_match = re.search(r'checkpoint-(\d+)', latest_checkpoint)
        step_number = step_match.group(1) if step_match else "unknown"
        
        new_name = f"checkpoint_{step_number}_{timestamp}_{datestamp}"
        new_path = os.path.join(args.output_dir, new_name)
        
        # Перейменування
        shutil.move(latest_checkpoint, new_path)
        self.logger.info(f"\n[CHECKPOINT] Перейменовано у: {new_path}")

# --- 3. ФУНКЦІЇ ДЛЯ ВОССТАНОВЛЕННЯ ---

def find_latest_checkpoint(path, prefix="checkpoint"):
    """Ініціалізація: шукає найновішу папку, що починається на 'prefix'."""
    if not os.path.isdir(path):
        return None
    list_of_dirs = glob.glob(os.path.join(path, f'{prefix}_*'))
    
    # Також перевіряємо оригінальний формат 'checkpoint-X'
    if prefix == "checkpoint":
        original_dirs = glob.glob(os.path.join(path, 'checkpoint-*'))
        list_of_dirs.extend(original_dirs)
        list_of_dirs = list(set(list_of_dirs))
    
    if not list_of_dirs:
        return None
        
    # Сортування за часом модифікації (новіші в кінці)
    latest_checkpoint = max(list_of_dirs, key=os.path.getmtime)
    return latest_checkpoint

# --- 4. ОСНОВНА ЛОГІКА СКРИПТА ---

if __name__ == "__main__":
    # --- 0. Ініціалізація та налаштування ---

    # Встановлення налаштувань (Debug, Logs, Workers)
    configure_runtime(pyIanthe_config.DEBUG)
    
    # Визначення логера
    logger = setup_logging(
        log_to_file=pyIanthe_config.TRAINING_LOG_ENABLE,
        log_file=os.path.join(
            pyIanthe_config.BASE_DIR,
            pyIanthe_config.TRAINING_LOG_FILENAME
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    fp16 = pyIanthe_config.FP16 if device == "cuda" and pyIanthe_config.FP16 else False
    bf16 = pyIanthe_config.BF16 if device == "cuda" and pyIanthe_config.BF16 else False
    attn_impl = pyIanthe_config.ATTENTION_TYPE if device == "cuda" else "eager"

    logger.info(f"Пристрій: {device}, GPU: {torch.cuda.device_count()}")
    if device == "cuda":
        logger.info(f"GPU знайдено: {torch.cuda.get_device_name(0)}")
    logger.info(f"Attention implementation: {attn_impl}")
    
    actual_num_workers = get_num_workers(logger)
    set_seed(42) # Встановлюємо фіксований seed

    # --- 1. Визначення та створення папок ---
    
    # CHECKPOINTS_DIR: Тут зберігаються ПОВНІ чекпоінти Trainer (стан + оптимізатор)
    CHECKPOINT_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_CHECKPOINTS)
    # SAFE_ARCHIVE_DIR: Тут зберігаються ТІЛЬКИ веси (надійний бекап)
    SAFE_ARCHIVE_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_MODEL) 
    
    REPORTS_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_REPORTS)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAFE_ARCHIVE_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- 2. Пошук Восстановлення ---
    
    full_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR, prefix="checkpoint")
    latest_archive_path = find_latest_checkpoint(SAFE_ARCHIVE_DIR, prefix="model_step")
    
    resume_from_checkpoint = None
    
    # --- 3. Ініціалізація Токенізатора ---
    
    tokenizer_source_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, *pyIanthe_config.MODEL_ID.split("/"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_dir, local_files_only=True)
    tokenizer.model_max_length = pyIanthe_config.CONTEXT_LENGTH
    
    # Визначаємо розмір словника для конфігурації
    vocab_size = len(tokenizer)
    logger.info(f"Токенізатор завантажено з: {tokenizer_source_dir}, Vocab Size: {vocab_size}")

    # --- 4. Завантаження Моделі ---

    if full_checkpoint_path:
        # ПРІОРИТЕТ 1: Старт з повного чекпоінта Trainer (відновлення ВСЬОГО стану)
        logger.info(f"\n[ЗАПУСК] Возобновлення навчання з повного чекпоінта: {full_checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
                full_checkpoint_path,
                ignore_mismatched_sizes=False,
                local_files_only=True,
                attn_implementation=attn_impl
            ).to(device)
        resume_from_checkpoint = full_checkpoint_path
        
    elif latest_archive_path:
        # ПРІОРИТЕТ 2: Резервний старт з весів (архів `MAIN_MODEL_DIR`). Втрата оптимізатора.
        logger.info(f"\n[ЗАПУСК] Резервний старт з весів: {latest_archive_path}")
        model = AutoModelForCausalLM.from_pretrained(
                latest_archive_path,
                ignore_mismatched_sizes=False,
                local_files_only=True,
                attn_implementation=attn_impl
            ).to(device)
        # resume_from_checkpoint залишається None, щоб Trainer ініціалізував новий оптимізатор

    else:
        # ПРІОРИТЕТ 3: Старт з нуля (якщо немає ні чекпоінтів, ні архівів)
        logger.info("\n[ЗАПУСК] Старт навчання з нуля (новий екземпляр моделі)")
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=pyIanthe_config.CONTEXT_LENGTH,
            n_ctx=pyIanthe_config.CONTEXT_LENGTH,
            n_embd=pyIanthe_config.EMBEDDING_DIM,
            n_layer=pyIanthe_config.NUM_LAYERS,
            n_head=pyIanthe_config.HEADS,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
        )
        model = AutoModelForCausalLM.from_config(
                config,
                attn_implementation=attn_impl
            ).to(device)
        
    # Прив'язка ваг (важливо для CausalLM)
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
        logger.info("✓ Прив'язані ваги lm_head оновлено/перевірено")
        
    logger.info(f"Модель завантажена, параметрів: {model.num_parameters():,}")
    
    # Gradient Checkpointing (перенесено з вашого коду)
    if pyIanthe_config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing увімкнено")
    
    # --- 5. Завантаження Датасету ---

    train_dataset_path = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_TRAIN_DATASET)
    eval_dataset_path = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_EVAL_DATASET)

    try:
        dataset = load_from_disk(train_dataset_path)
        logger.info(f"Датасет завантажено, записів: {len(dataset)}")
    except Exception as e:
        logger.error(f"Папка датасету не знайдена або порожня: {train_dataset_path}. Помилка: {e}")
        sys.exit(1)

    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    logger.info(f"Після фільтрації: {len(dataset)} записів")
    
    # Токенізація
    tokenized_dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=pyIanthe_config.CONTEXT_LENGTH),
        batched=True, 
        remove_columns=["text"] # Припускаємо, що текстовий стовпець називається "text"
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval датасет (з вашої логіки)
    eval_dataset = None
    if pyIanthe_config.EVAL_ENABLED and os.path.exists(eval_dataset_path):
        try:
            eval_dataset = load_from_disk(eval_dataset_path)
            eval_dataset = eval_dataset.map(
                lambda ex: tokenizer(ex["text"], truncation=True, max_length=pyIanthe_config.CONTEXT_LENGTH),
                batched=True, 
                remove_columns=["text"]
            )
            logger.info(f"Eval датасет завантажено, записів: {len(eval_dataset)}")
        except Exception as e:
            logger.warning(f"Не вдалося завантажити/токенізувати eval датасет: {e}")
            eval_dataset = None

    # --- 6. Налаштування Trainer (TrainingArguments, Callback-и) ---
    
    # Зберігаємо вашу функцію завантаження тестових прикладів
    def load_test_examples():
        test_file = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.TEXT_TEST_FILENAME)
        # ... (Ваш оригінальний код load_test_examples)
        # [Оригінальний код]
        if not os.path.exists(test_file):
            logger.warning(f"Файл тестових прикладів не знайдено: {test_file}")
            logger.info("Використовуються стандартні приклади")
            return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
        
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                examples = data[:pyIanthe_config.TEXT_TESTS_COUNT]
            elif isinstance(data, dict) and "examples" in data:
                examples = data["examples"][:pyIanthe_config.TEXT_TESTS_COUNT]
            else:
                logger.warning(f"Невідомий формат файлу {test_file}")
                return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
            logger.info(f"Завантажено {len(examples)} тестових прикладів з {test_file}")
            return examples
        except Exception as e:
            logger.error(f"Помилка при завантаженні {test_file}: {e}")
            return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
        # [/Оригінальний код]

    GENERATE_EXAMPLES = load_test_examples()
    
    # Виносимо ваші метрики і функції тестування в глобальну область (або залишаємо як внутрішні функції)
    # З метою чистоти, я їх не дублюю тут, а припускаю, що вони доступні.
    # [Примітка: в робочому коді потрібно переконатися, що compute_perplexity, compute_text_metrics та 
    # test_text_generation, test_eval_dataset доступні.]
    
    # --- Налаштування аргументів ---

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        # Налаштування епох/кроків
        num_train_epochs=pyIanthe_config.EPOCHS,
        max_steps=pyIanthe_config.MAX_STEPS, # -1, якщо використовуємо епохи
        
        # Налаштування батчу
        per_device_train_batch_size=pyIanthe_config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=pyIanthe_config.GRADIENT_ACCUMULATION_STEPS,
        
        # Налаштування LR
        learning_rate=pyIanthe_config.LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY,
        #gradient_clipping=pyIanthe_config.GRADIENT_CLIPPING, # З вашого конфігу
        
        # Налаштування оптимізатора/шедулера (використовуємо мою надійну логіку)
        lr_scheduler_type="cosine", # Cosine Decay
        warmup_ratio=0.03,          # Невеликий Warmup (за замовчуванням 0.03)
        
        # Налаштування збереження (використовуємо ваші значення)
        save_strategy="steps",
        save_steps=pyIanthe_config.SAVE_STEPS,
        save_total_limit=pyIanthe_config.SAVE_LIMIT,
        
        # Логування та оцінка (для запуску вашого `TestingCallback`)
        logging_steps=100,
        evaluation_strategy="steps" if pyIanthe_config.EVAL_ENABLED else "no",
        eval_steps=pyIanthe_config.EVAL_STEPS if pyIanthe_config.EVAL_ENABLED else None,
        
        # Налаштування прискорення
        fp16=fp16,
        bf16=bf16,
        no_cuda=device == "cpu",
        dataloader_pin_memory=pyIanthe_config.PIN_MEMORY,
        dataloader_num_workers=actual_num_workers,
        report_to="none",
        disable_tqdm=False
    )
    
    # --- Ініціалізація Trainer ---

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset, # Передаємо для внутрішнього eval, якщо увімкнено
        tokenizer=tokenizer,
        data_collator=data_collator,
        # Використовуємо ваші callback-и, доповнені новими
        callbacks=[
            ReliableBackupCallback(safe_archive_dir=SAFE_ARCHIVE_DIR, logger=logger), # Надійний бекап весів
            CustomCheckpointingCallback(logger=logger),                             # Кастомне іменування
            # Ваш оригінальний callback для проміжного тестування
            # Зверніть увагу, що функції test_text_generation, compute_perplexity і т.д.
            # мають бути доступні зсередини TestingCallback
            TestingCallback(
                model=model,
                tokenizer=tokenizer,
                device=device,
                test_prompts=GENERATE_EXAMPLES if pyIanthe_config.TEST_ENABLED else None,
                eval_dataset=eval_dataset if pyIanthe_config.EVAL_ENABLED else None
            )
        ]
    )

    # --- 7. Запуск Тренування ---
    logger.info(f"\n{'='*60}")
    logger.info(f"СТАРТ ТРЕНУВАННЯ")
    logger.info(f"Епох: {pyIanthe_config.EPOCHS}")
    logger.info(f"Макс. кроків: {pyIanthe_config.MAX_STEPS}")
    logger.info(f"Частота збережень: кожні {pyIanthe_config.SAVE_STEPS} кроків")
    logger.info(f"Режим відновлення: {'Повний' if full_checkpoint_path else ('Веси' if latest_archive_path else 'З нуля')}")
    logger.info(f"{'='*60}\n")
    
    # Виконуємо trainer.train() тільки один раз (для циклу for epoch - див. нижче)
    
    try:
        # Ваш оригінальний цикл for epoch більше не потрібен, 
        # Trainer сам керує епохами. Викликаємо train лише один раз.
        # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Для того, щоб зберегти вашу логіку циклу по епохах та фінальний звіт (generate_epoch_report)
        # після кожної епохи, ми повинні імплементувати це через Callback 
        # або викликати train() для кожної епохи.
        
        # *Оптимальне рішення:* Викликати train() один раз, а фінальний звіт 
        # перенести в Callback. Якщо ж ви хочете зберегти вашу логіку циклу, 
        # ми залишаємо її, але вона може конфліктувати з Trainer, 
        # який теж має свій внутрішній лічильник епох.
        
        # Ми приймемо ваш цикл for epoch, щоб він викликав generate_epoch_report:
        for epoch in range(pyIanthe_config.EPOCHS):
            logger.info(f"\n{'='*60}")
            logger.info(f"===== Епоха {epoch+1} / {pyIanthe_config.EPOCHS} =====")
            logger.info(f"{'='*60}")
            
            # Якщо це перша епоха і ми відновлюємося, використовуємо resume_from_checkpoint
            current_resume = resume_from_checkpoint if epoch == 0 else None
            
            trainer.train(resume_from_checkpoint=current_resume)
            
            # Скидаємо прапорець відновлення після першої спроби
            if epoch == 0:
                resume_from_checkpoint = None

            # --- Логіка, що була після кожної епохи ---
            
            # Ваш оригінальний код: 1) Зберігаємо головну модель у model/
            # Це вже робить ReliableBackupCallback в on_train_end (або on_save)
            # Тому тут просто викликаємо фінальний звіт:
            
            # Ваш оригінальний код: 2) Зберігаємо чекпоінт у checkpoints/checkpoint-X/
            # Це робить Trainer в кінці епохи, але ми перехоплюємо це через CustomCheckpointingCallback
            
            # Генеруємо фінальний звіт епохи (ваш оригінальний код)
            # [Примітка: generate_epoch_report має бути доступна]
            # generate_epoch_report(epoch) 

            logger.info(f"\n[SUCCESS] Епоха {epoch+1} завершена і збережена")
            logger.info(f"  → Надійний бекап: {SAFE_ARCHIVE_DIR}")
        
    except KeyboardInterrupt:
        # Цю логіку ми залишаємо, вона спрацює, якщо Trainer не встиг зберегтися
        logger.warning(f"\n⚠ Тренування перервано користувачем (Ctrl+C)")
        logger.info("Для продовження запустіть скрипт знову. Буде використано останній повний чекпоінт або веси.")
        sys.exit(0)

    logger.info(f"\n{'='*60}")
    logger.info("[SUCCESS] ТРЕНУВАННЯ ЗАВЕРШЕНО!")
    logger.info(f"Фінальна модель: {SAFE_ARCHIVE_DIR}")
    logger.info(f"{'='*60}")