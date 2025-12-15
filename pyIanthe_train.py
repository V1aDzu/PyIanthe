# pyIanthe_train_final.py
import os
import sys
import json
import torch
import shutil
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config,
    TrainerCallback
)
from datasets import load_from_disk
from modules.pyIanthe_log import setup_logging, configure_runtime
from modules.pyIanthe_hw import get_num_workers
import pyIanthe_config

if __name__ == "__main__":
    # 0. Включаємо Debug повідомлення та ведення логів

    configure_runtime(pyIanthe_config.DEBUG)
    
    logger = setup_logging(
        log_to_file=pyIanthe_config.TRAINING_LOG_ENABLE,
        log_file=os.path.join(
            pyIanthe_config.BASE_DIR,
            pyIanthe_config.TRAINING_LOG_FILENAME
        )
    )

    # 0. GPU та налаштування
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    fp16 = pyIanthe_config.FP16 if device == "cuda" else False
    bf16 = pyIanthe_config.BF16 if device == "cuda" else False
    pin_memory = pyIanthe_config.PIN_MEMORY if device == "cuda" else False
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Attention type
    attn_impl = pyIanthe_config.ATTENTION_TYPE if device == "cuda" else "eager"
    logger.info(f"Attention implementation: {attn_impl}")

    logger.info(f"Пристрій: {device}, GPU: {num_gpus}")
    if device == "cuda":
        logger.info(f"GPU знайдено: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Тренування буде на CPU")

    # Визначаємо фактичну кількість воркерів
    actual_num_workers = get_num_workers(
        config=pyIanthe_config,
        logger=logger
    )

    # 1. Папки
    CHECKPOINT_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_CHECKPOINTS)
    MAIN_MODEL_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_MODEL)
    REPORTS_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_REPORTS)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MAIN_MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Функція для останнього чекпоінта
    def get_last_checkpoint():
        if not os.path.isdir(CHECKPOINT_DIR):
            return None
        dirs = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
        if not dirs:
            return None
        try:
            checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                dirs_sorted = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                return os.path.join(CHECKPOINT_DIR, dirs_sorted[-1])
        except:
            pass
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

    # 3. Завантаження датасету
    train_dataset_path = pyIanthe_config.FOLDER_TRAIN_DATASET
    if not os.path.exists(train_dataset_path):
        logger.error(f"Папка датасету не знайдена: {train_dataset_path}")
        sys.exit(1)

    arrow_files = [f for f in os.listdir(train_dataset_path) if f.endswith('.arrow')]
    dataset_info = os.path.join(train_dataset_path, 'dataset_info.json')
    if not arrow_files and not os.path.exists(dataset_info):
        logger.error(f"Папка {train_dataset_path} порожня або не містить датасет")
        sys.exit(1)

    dataset = load_from_disk(train_dataset_path)
    logger.info(f"Датасет завантажено, записів: {len(dataset)}")

    # Завантаження eval датасету якщо потрібно
    eval_dataset = None
    if pyIanthe_config.EVAL_ENABLED:
        eval_dataset_path = pyIanthe_config.FOLDER_EVAL_DATASET
        if os.path.exists(eval_dataset_path):
            try:
                eval_dataset = load_from_disk(eval_dataset_path)
                logger.info(f"Eval датасет завантажено, записів: {len(eval_dataset)}")
            except Exception as e:
                logger.warning(f"Не вдалося завантажити eval датасет: {e}")

    # 4. Токенізатор
    tokenizer_source_dir = os.path.join(pyIanthe_config.FOLDER_MODELS, *pyIanthe_config.MODEL_ID.split("/"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_dir, local_files_only=True)
    tokenizer.model_max_length = CONTEXT_LENGTH
    logger.info(f"Токенізатор завантажено з: {tokenizer_source_dir}")

    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    logger.info(f"Після фільтрації: {len(dataset)} записів")
    
    tokenized_dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=CONTEXT_LENGTH),
                                    batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Завантаження моделі
    vocab_size = len(tokenizer)
    checkpoint_has_model = last_checkpoint and os.path.exists(os.path.join(last_checkpoint, "model.safetensors"))
    main_model_exists = os.path.exists(os.path.join(MAIN_MODEL_DIR, "model.safetensors"))

    if checkpoint_has_model:
        logger.info(f"Знайдено чекпоінт з моделлю: {last_checkpoint}")
        logger.info("Завантажуємо модель з чекпоінта (навчання продовжиться)")
        model = AutoModelForCausalLM.from_pretrained(
                last_checkpoint,
                ignore_mismatched_sizes=False,
                local_files_only=True,
                attn_implementation=attn_impl  # аттеншен
            ).to(device)
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            logger.info("✓ Прив'язані ваги lm_head оновлено")
        resume_from = last_checkpoint
        
    elif main_model_exists:
        logger.info(f"Чекпоінта немає, але знайдено модель у: {MAIN_MODEL_DIR}")
        logger.info("Завантажуємо модель (optimizer почнеться з нуля)")
        model = AutoModelForCausalLM.from_pretrained(
                MAIN_MODEL_DIR,
                ignore_mismatched_sizes=False,
                local_files_only=True,
                attn_implementation=attn_impl  # аттеншен
            ).to(device)
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            logger.info("✓ Прив'язані ваги lm_head оновлено")
        resume_from = None
        
    else:
        logger.info("Не знайдено ні чекпоінта, ні моделі")
        logger.info("Створюємо нову модель з конфігурації GPT2")
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
            loss_type=None,
        )
        model = AutoModelForCausalLM.from_config(
                config,
                attn_implementation=attn_impl  # аттеншен
            ).to(device)
        resume_from = None

    logger.info(f"Модель завантажена, параметрів: {model.num_parameters():,}")
    
    # Перевірка привязки lm_head до embeddings
    if hasattr(model, 'lm_head') and hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'wte'):
            lm_head_ptr = model.lm_head.weight.data_ptr()
            wte_ptr = model.transformer.wte.weight.data_ptr()
            if lm_head_ptr == wte_ptr:
                logger.info("✓ lm_head.weight правильно прив'язаний до wte")
            else:
                logger.warning("⚠ lm_head.weight НЕ прив'язаний до wte")

    # Gradient Checkpointing
    if pyIanthe_config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing увімкнено")
        logger.info("  → Можна збільшити batch size")
    # 6. Завантаження тестових прикладів
    def load_test_examples():
        """Завантажує тестові приклади з JSON файлу"""
        test_file = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.TEXT_TEST_FILENAME)
        
        if not os.path.exists(test_file):
            logger.warning(f"Файл тестових прикладів не знайдено: {test_file}")
            logger.info("Використовуються стандартні приклади")
            return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
        
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, list):
                examples = data[:pyIanthe_config.TEXT_TESTS_COUNT]
                logger.info(f"Завантажено {len(examples)} тестових прикладів з {test_file}")
                return examples
                
            elif isinstance(data, dict) and "examples" in data:
                examples = data["examples"][:pyIanthe_config.TEXT_TESTS_COUNT]
                logger.info(f"Завантажено {len(examples)} тестових прикладів з {test_file}")
                return examples
                
            else:
                logger.warning(f"Невідомий формат файлу {test_file}")
                logger.info("Використовуються стандартні приклади")
                return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
                
        except Exception as e:
            logger.error(f"Помилка при завантаженні {test_file}: {e}")
            logger.info("Використовуються стандартні приклади")
            return ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]

    GENERATE_EXAMPLES = load_test_examples()

    # 7. Функції тестування та метрик
    def compute_perplexity(logits, labels):
        """Обчислює perplexity"""
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return torch.exp(loss.mean()).item()

    def compute_text_metrics(text):
        """Обчислює метрики тексту"""
        tokens = text.split()
        total_tokens = len(tokens)
        meaningful_tokens = sum(1 for t in tokens if t.isalnum())
        perc_meaningful = meaningful_tokens / total_tokens * 100 if total_tokens else 0
        return total_tokens, perc_meaningful

    def test_text_generation(model, tokenizer, prompts, device):
        """Тестує генерацію тексту на прикладах"""
        model.eval()
        results = {}
        
        generation_config = {
            "max_new_tokens": TRAINING_CONFIG["max_new_tokens"],
            "temperature": TRAINING_CONFIG["temperature"],
            "top_p": TRAINING_CONFIG["top_p"],
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Генерація
                outputs = model.generate(**inputs, **generation_config)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Метрики
                logits = model(**inputs).logits
                ppl = compute_perplexity(logits, inputs["input_ids"])
                total_tokens, perc_meaningful = compute_text_metrics(generated_text)
                
                results[prompt] = {
                    "generated_text": generated_text,
                    "perplexity": ppl,
                    "total_tokens": total_tokens,
                    "perc_meaningful_tokens": perc_meaningful
                }
        
        model.train()
        return results

    def test_eval_dataset(model, tokenizer, eval_dataset, device, max_samples):
        """Тестує модель на eval датасеті"""
        if eval_dataset is None:
            return None
            
        model.eval()
        
        # Обмежуємо кількість зразків
        test_samples = min(max_samples, len(eval_dataset))
        eval_subset = eval_dataset.select(range(test_samples))
        
        total_loss = 0
        total_perplexity = 0
        num_samples = 0
        
        with torch.no_grad():
            for sample in eval_subset:
                try:
                    text = sample.get("text", "")
                    if not text or not isinstance(text, str):
                        continue
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                     max_length=CONTEXT_LENGTH).to(device)
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    ppl = torch.exp(outputs.loss).item()
                    
                    total_loss += loss
                    total_perplexity += ppl
                    num_samples += 1
                    
                except Exception as e:
                    if pyIanthe_config.DEBUG:
                        logger.warning(f"Помилка обробки зразка: {e}")
                    continue
        
        model.train()
        
        if num_samples == 0:
            return None
        
        return {
            "num_samples": num_samples,
            "avg_loss": total_loss / num_samples,
            "avg_perplexity": total_perplexity / num_samples
        }

    # 8. Callback для промежуточного тестування
    class TestingCallback(TrainerCallback):
        """Callback для запуску тестів під час тренування"""
        
        def __init__(self, model, tokenizer, device, test_prompts, eval_dataset):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.test_prompts = test_prompts
            self.eval_dataset = eval_dataset
            self.current_epoch = 0
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            """Відслідковує початок епохи"""
            self.current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        def on_step_end(self, args, state, control, **kwargs):
            """Викликається після кожного кроку"""
            # Перевіряємо чи потрібно запускати тести
            if state.global_step % pyIanthe_config.EVAL_STEPS != 0:
                return
            
            if state.global_step == 0:
                return
            
            # Якщо обидва тести вимкнені - не запускати callback взагалі
            if not pyIanthe_config.TEST_ENABLED and not pyIanthe_config.EVAL_ENABLED:
                return
            
            # Запускаємо тести
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ПРОМІЖНЕ ТЕСТУВАННЯ")
            logger.info(f"Епоха: {self.current_epoch}, Крок: {state.global_step}")
            logger.info(f"Час: {timestamp}")
            logger.info(f"{'='*60}")
            
            test_results = {
                "timestamp": timestamp,
                "epoch": self.current_epoch,
                "global_step": state.global_step,
                "text_generation": None,
                "eval_dataset": None
            }
            
            # Тест генерації тексту
            if pyIanthe_config.TEST_ENABLED and self.test_prompts:
                logger.info("\n[TEST] Тестування генерації тексту...")
                text_results = test_text_generation(
                    self.model, self.tokenizer, self.test_prompts, self.device
                )
                test_results["text_generation"] = text_results
                
                # Виводимо приклад
                first_prompt = self.test_prompts[0]
                result = text_results[first_prompt]
                logger.info(f"  Промпт: {first_prompt}")
                logger.info(f"  Текст: {result['generated_text'][:100]}...")
                logger.info(f"  Perplexity: {result['perplexity']:.2f}")
                logger.info(f"  Токенів: {result['total_tokens']}")
            
            # Тест на eval датасеті
            if pyIanthe_config.EVAL_ENABLED and self.eval_dataset:
                logger.info("\n[EVAL] Тестування на eval датасеті...")
                eval_results = test_eval_dataset(
                    self.model, self.tokenizer, self.eval_dataset, 
                    self.device, pyIanthe_config.EVAL_TESTS_COUNT
                )
                if eval_results:
                    test_results["eval_dataset"] = eval_results
                    logger.info(f"  Зразків: {eval_results['num_samples']}")
                    logger.info(f"  Avg Loss: {eval_results['avg_loss']:.4f}")
                    logger.info(f"  Avg Perplexity: {eval_results['avg_perplexity']:.2f}")
            
            # Зберігаємо JSON звіт тільки якщо є результати
            if test_results["text_generation"] or test_results["eval_dataset"]:
                report_filename = f"test_epoch{self.current_epoch}_step{state.global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_path = os.path.join(REPORTS_DIR, report_filename)
                
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"\n✓ Звіт збережено: {report_filename}")
            else:
                logger.info(f"\n[INFO] Тести вимкнені, звіт не створено")
            
            logger.info(f"{'='*60}\n")

    # 9. TrainingArguments і Trainer
    
    # Розрахунок ефективного розміру батчу
    effective_batch_size = (PER_DEVICE_BATCH_SIZE * 
                           pyIanthe_config.GRADIENT_ACCUMULATION_STEPS * 
                           max(1, num_gpus))
    
    logger.info(f"\n[BATCH CONFIG]")
    logger.info(f"  Per-device batch size: {PER_DEVICE_BATCH_SIZE}")
    logger.info(f"  Gradient accumulation steps: {pyIanthe_config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Number of GPUs: {max(1, num_gpus)}")
    logger.info(f"  Num workers: {actual_num_workers}")
    logger.info(f"  → Effective batch size: {effective_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=pyIanthe_config.GRADIENT_ACCUMULATION_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=pyIanthe_config.SAVE_LIMIT,
        logging_steps=100,  # Частіше логувати для моніторингу
        learning_rate=LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY,
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=actual_num_workers,
        report_to="none",
        disable_tqdm=False
    )

    # Створюємо callback
    testing_callback = TestingCallback(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_prompts=GENERATE_EXAMPLES if pyIanthe_config.TEST_ENABLED else None,
        eval_dataset=eval_dataset if pyIanthe_config.EVAL_ENABLED else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[testing_callback]  # Додаємо callback
    )

    # 10. Функція для фінального звіту після епохи
    def generate_epoch_report(epoch_index):
        """Генерує детальний звіт після завершення епохи"""
        model.eval()
        report = {}
        
        logger.info(f"\n[REPORT] Генерація звіту по епосі {epoch_index+1}...")
        
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
                avg_len, perc_meaningful = compute_text_metrics(text)
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
        
        logger.info(f"✓ Звіт по епосі {epoch_index+1} збережено: {report_file}")
        model.train()

    # 11. Тренування
    logger.info(f"\n{'='*60}")
    logger.info(f"СТАРТ ТРЕНУВАННЯ")
    logger.info(f"Епох: {EPOCHS}")
    logger.info(f"Тестування: {'Увімкнено' if pyIanthe_config.TEST_ENABLED else 'Вимкнено'}")
    logger.info(f"Eval: {'Увімкнено' if pyIanthe_config.EVAL_ENABLED else 'Вимкнено'}")
    if pyIanthe_config.TEST_ENABLED or pyIanthe_config.EVAL_ENABLED:
        logger.info(f"Частота тестів: кожні {pyIanthe_config.EVAL_STEPS} кроків")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(EPOCHS):
        logger.info(f"\n{'='*60}")
        logger.info(f"===== Епоха {epoch+1} / {EPOCHS} =====")
        logger.info(f"{'='*60}")
        
        # Оновлюємо епоху в callback
        testing_callback.current_epoch = epoch + 1
        
        try:
            # Тренуємо 1 епоху
            if epoch == 0 and resume_from:
                logger.info(f"Продовжуємо з чекпоінта: {resume_from}")
                trainer.train(resume_from_checkpoint=resume_from)
            else:
                trainer.train()
                
        except KeyboardInterrupt:
            logger.warning(f"\n⚠ Тренування перервано користувачем (Ctrl+C)")
            logger.info("Зберігаємо поточний стан...")
            
            # Зберігаємо головну модель
            if hasattr(model.config, 'tie_word_embeddings'):
                model.config.tie_word_embeddings = True
            model.save_pretrained(MAIN_MODEL_DIR)
            tokenizer.save_pretrained(MAIN_MODEL_DIR)
            logger.info(f"✓ Модель збережена у: {MAIN_MODEL_DIR}")
            
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
            
            logger.info(f"✓ Аварійний чекпоінт: {emergency_checkpoint}")
            logger.info("Для продовження запустіть скрипт знову")
            sys.exit(0)
        
        # Після кожної епохи зберігаємо:
        
        # 1) Головну модель у model/
        logger.info(f"\nЗберігаємо головну модель у: {MAIN_MODEL_DIR}")
        if hasattr(model.config, 'tie_word_embeddings'):
            model.config.tie_word_embeddings = True
        model.save_pretrained(MAIN_MODEL_DIR)
        tokenizer.save_pretrained(MAIN_MODEL_DIR)
        
        # 2) Чекпоінт у checkpoints/checkpoint-X/
        checkpoint_epoch_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch+1}")
        logger.info(f"Зберігаємо чекпоінт у: {checkpoint_epoch_dir}")
        os.makedirs(checkpoint_epoch_dir, exist_ok=True)
        
        if hasattr(model.config, 'tie_word_embeddings'):
            model.config.tie_word_embeddings = True
        trainer.save_model(checkpoint_epoch_dir)
        trainer.save_state()
        
        # Копіюємо файли стану у checkpoint
        for fname in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"]:
            src = os.path.join(CHECKPOINT_DIR, fname)
            dst = os.path.join(checkpoint_epoch_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Генеруємо фінальний звіт епохи
        generate_epoch_report(epoch)
        
        logger.info(f"\n[SUCCESS] Епоха {epoch+1} завершена і збережена")
        logger.info(f"  → Модель: {MAIN_MODEL_DIR}")
        logger.info(f"  → Чекпоінт: {checkpoint_epoch_dir}")

    logger.info(f"\n{'='*60}")
    logger.info("[SUCCESS] ТРЕНУВАННЯ ЗАВЕРШЕНО!")
    logger.info(f"Всього епох: {EPOCHS}")
    logger.info(f"Фінальна модель: {MAIN_MODEL_DIR}")
    logger.info(f"Звіти: {REPORTS_DIR}")
    logger.info(f"Лог: {pyIanthe_config.TRAINING_LOG_FILENAME}")
    logger.info(f"{'='*60}")
