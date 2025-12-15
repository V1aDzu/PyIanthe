# pyIanthe_train_final.py
import os
import sys
import json
import torch
import shutil
import numpy as np
import random

import hashlib

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config    
)
from modules.pyIanthe_log import setup_logging, configure_runtime
from modules.pyIanthe_hw import get_device_config, get_num_workers
from modules.pyIanthe_utils import get_last_checkpoint, load_test_examples
from modules.pyIanthe_dataset import load_datasets
from modules.pyIanthe_metrics import compute_perplexity, compute_text_metrics
from modules.pyIanthe_test_callbacks import TestingCallback
import pyIanthe_config

from transformers.trainer_callback import TrainerState


if __name__ == "__main__":
    # 0. Включаємо Debug повідомлення та ведення логів



    def save_all_rng_states(filename):
        """Сохраняет состояния RNG всех необходимых библиотек в один файл."""
        
        # Собираем все состояния в словарь
        full_state = {
            'python_rng_state': random.getstate(),        # Стандартный 'random'
            'numpy_rng_state': np.random.get_state(),      # NumPy
            'torch_cpu_rng_state': torch.get_rng_state(),  # PyTorch CPU (для DataLoader)
            # PyTorch CUDA (для весов, Dropout и т.д.)
            'torch_cuda_rng_states': torch.cuda.get_rng_state_all(), 
        }
        
        # Сохраняем словарь
        torch.save(full_state, filename)
        print(f"✅ Полные состояния RNG сохранены в {filename}")

    def load_all_rng_states(filename):
        """Загружает и восстанавливает состояния RNG из файла."""
        if not os.path.exists(filename):
            print(f"❌ Ошибка: Файл состояний {filename} не найден.")
            return
            
        full_state = torch.load(filename, weights_only=False)
        
        # 1. Восстанавливаем Python random
        random.setstate(full_state['python_rng_state'])
        
        # 2. Восстанавливаем NumPy
        np.random.set_state(full_state['numpy_rng_state'])
        
        # 3. Восстанавливаем PyTorch CPU
        torch.set_rng_state(full_state['torch_cpu_rng_state'])
        
        # 4. Восстанавливаем PyTorch CUDA (самое важное для LLM)
        torch.cuda.set_rng_state_all(full_state['torch_cuda_rng_states'])
        
        print(f"✅ Полные состояния RNG успешно восстановлены из {filename}")


    #def save_rng_to_file(filename="cuda_rng_state.pt"):
     #   # Получаем список тензоров состояний для всех GPU
     #   rng_states0 = random.getstate()
     #   rng_states1 = np.random.get_state()
     #   rng_states2 = torch.cuda.get_rng_state_all()
        # Сохраняем как обычный объект PyTorch
      ##  torch.save(rng_states0, "cuda0_rng.pt")
       # torch.save(rng_states1, "cuda1_rng.pt")
       ## torch.save(rng_states2, "cuda2_rng.pt")
        #print(f"RNG состояния всех GPU сохранены в файл: {filename}")

##    "python": random.getstate(),
             #   "numpy": np.random.get_state(),
             #   "cpu": torch.random.get_rng_state(),







#    def load_rng_from_file(filename="cuda_rng_state.pt"):
 #       try:
  #          rng_states0 = torch.load("cuda0_rng.pt")
   #         rng_states1 = torch.load("cuda1_rng.pt")
    #        rng_states2 = torch.load("cuda2_rng.pt")
     #       random.set_rng_state_all(rng_states0)
      #      torch.get_rng_state(), random.getstate() и numpy.random.get_state().
       #     torch.cuda.set_rng_state_all(rng_states1)
        #    torch.cuda.set_rng_state_all(rng_states2)

       # rng_states0 = random.getstate()
        #rng_states1 = np.random.get_state()
        #rng_states2 = torch.cuda.get_rng_state_all()
        # Сохраняем как обычный объект PyTorch
       # torch.save(rng_states0, "cuda0_rng.pt")
       # torch.save(rng_states1, "cuda1_rng.pt")
       # torch.save(rng_states2, "cuda2_rng.pt")



    configure_runtime(pyIanthe_config.DEBUG)
    
    logger = setup_logging(
        log_to_file=pyIanthe_config.TRAINING_LOG_ENABLE,
        log_file=os.path.join(
            pyIanthe_config.BASE_DIR,
            pyIanthe_config.TRAINING_LOG_FILENAME
        )
    )

    # 0. GPU та налаштування
    device, fp16, bf16, pin_memory, num_gpus, attn_impl = get_device_config(
        config=pyIanthe_config,
        logger=logger       
    )    





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
    LAST_CHECKPOINT = get_last_checkpoint(CHECKPOINT_DIR)

    # 2. Конфігурація навчання
    EPOCHS = pyIanthe_config.EPOCHS
    LEARNING_RATE = pyIanthe_config.LEARNING_RATE
    PER_DEVICE_BATCH_SIZE = pyIanthe_config.PER_DEVICE_BATCH_SIZE
    CONTEXT_LENGTH = pyIanthe_config.CONTEXT_LENGTH
    SAVE_STEPS = pyIanthe_config.SAVE_STEPS
    MAX_NEW_TOKENS = pyIanthe_config.GEN_TEST_MNEW_TOKENS
    TEMPERATURE = pyIanthe_config.GEN_TEST_TEMPERATURE
    TOP_P = pyIanthe_config.GEN_TEST_TOP_P

    # Конфігурація моделі
    N_EMBD = pyIanthe_config.EMBEDDING_DIM
    N_LAYER = pyIanthe_config.NUM_LAYERS
    N_HEAD = pyIanthe_config.HEADS

    # 3. Завантаження датасету
    try:
        dataset, eval_dataset = load_datasets(
            config=pyIanthe_config,
            logger=logger
        )
    except Exception:
        sys.exit(1)

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
    main_model_exists = os.path.exists(os.path.join(MAIN_MODEL_DIR, "model.safetensors"))

    # Загружаем основную модель или создаём новую
    if main_model_exists:
        logger.info(f"Завантажуємо основну модель з {MAIN_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL_DIR,
            ignore_mismatched_sizes=False,
            local_files_only=True,
            attn_implementation=attn_impl
        ).to(device)
    else:
        logger.info("Основна модель не знайдена, створюємо нову")
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=CONTEXT_LENGTH,
            n_ctx=CONTEXT_LENGTH,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
            loss_type=None,
        )
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attn_impl
        ).to(device)

    # Привязка lm_head
    #if hasattr(model, 'tie_weights'):
    #    model.tie_weights()
    #    logger.info("✓ Прив'язані ваги lm_head оновлено")

    # 6. Завантаження тестових прикладів
    GENERATE_EXAMPLES = load_test_examples(
        logger=logger,
        base_dir=pyIanthe_config.BASE_DIR,
        filename=pyIanthe_config.TEXT_TEST_FILENAME,
        max_examples=pyIanthe_config.TEXT_TESTS_COUNT
    )

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
        eval_dataset=eval_dataset if pyIanthe_config.EVAL_ENABLED else None,
        config=pyIanthe_config,
        logger=logger
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[testing_callback]  # Додаємо callback
    )









    # Чекпоинт используется только для восстановления состояния тренера
    resume_from = LAST_CHECKPOINT if LAST_CHECKPOINT and os.path.exists(os.path.join(LAST_CHECKPOINT, "model.safetensors")) else None

    if resume_from:
        logger.info(f"Чекпоінт для відновлення тренера знайдено: {resume_from}")

        # Пути к аварийным файлам
        emerg_rng_state_path = os.path.join(resume_from, "rng_state.pth")
        emerg_scaler_state_path = os.path.join(resume_from, "scaler.pt")

        # Восстановление состояния RNG
        #if os.path.exists(emerg_rng_state_path):
         #   with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray]):

          #      rng_states = torch.load('rngstate.test')
           #     torch.cuda.set_rng_state_all(rng_states)
                # 2. Восстанавливаем и проверяем
            #    load_rng_from_file("my_rng.pt")
             #   print_rng_debug_info("ПОСЛЕ ЗАГРУЗКИ")


    

        load_all_rng_states("rngsave.txt")


        emerg_tr_state_path = os.path.join(resume_from, "trainer_state.json")
        emerg_opt_state_path = os.path.join(resume_from, "optimizer.pt")
        emerg_sched_state_path = os.path.join(resume_from, "scheduler.pt")
        emerg_rng_state_path = os.path.join(resume_from, "rng_state.pth")
        emerg_scaler_state_path = os.path.join(resume_from, "scaler.pt")




        #with open(emerg_rng_state_path, "r") as f:
        #    state_dict = json.load(f)
            
            # Створюємо об'єкт TrainerState з даних
        #    loaded_state = TrainerState(**state_dict)
            
        #    trainer.state = loaded_state

        full_state = torch.load(emerg_rng_state_path, weights_only=False) 
        trainer.state = full_state
# Далі йде load_all_rng_states, як ми робили

    
        if trainer.optimizer is not None and os.path.exists(emerg_opt_state_path):
            try:
                # Завантажуємо стан оптимізатора
                optimizer_state = torch.load(emerg_opt_state_path)
                trainer.optimizer.load_state_dict(optimizer_state)

                print(f"✅ Optimizer state відновлено з {emerg_opt_state_path}")
            except Exception as e:
                print(f"❌ Помилка відновлення стану оптимізатора: {e}")

    
        if trainer.lr_scheduler is not None and os.path.exists(emerg_sched_state_path):
            try:
                # Завантажуємо стан планувальника
                scheduler_state = torch.load(emerg_sched_state_path)
                trainer.lr_scheduler.load_state_dict(scheduler_state)
                print(f"✅ Scheduler state відновлено з {emerg_sched_state_path}")
            except Exception as e:
                print(f"❌ Помилка відновлення стану планувальника: {e}")

# --- ВИКОНАННЯ ---
# load_emergency_states(trainer, emerg_tr_state_path, emerg_opt_state_path, emerg_sched_state_path)

# Після цього ви можете викликати trainer.train()













              #  rng_states = torch.load(emerg_rng_state_path, weights_only=False)
              #  random.setstate(rng_states["python"])
              #  np.random.set_state(rng_states["numpy"])
              #  torch.random.set_rng_state(rng_states["cpu"])
              #  if "cuda" in rng_states and torch.cuda.is_available():
                    #torch.cuda.random.set_rng_state_all(rng_states["cuda"])
                    #cuda_states = rng_states.get("cuda", [])

                

              #      state_bytes = cuda_states.numpy().tobytes()
              #      state_hash = hashlib.md5(state_bytes).hexdigest()
            
                    # Печатаем ID устройства, размер состояния и хеш
               #     print(f"LOAD: Size={cuda_states.shape[0]} bytes | Hash={state_hash}")
        


                #    debug_cuda_rng_state(cuda_states, logger, prefix="[LOAD]")




                    #if cuda_states is not None and len(cuda_states) > 0:
                       # for i, state in enumerate(cuda_states):
                            #if i < num_gpus:
                             #   try:
                              #      torch.cuda.set_rng_state(state, device=i)
                              #  except RuntimeError as e:
                               #     print(f"GPU {i}: невозможно восстановить RNG ({e})")

              

                               
        logger.info("✓ RNG state відновлено з аварійного чекпоінта")

        if hasattr(model, 'lm_head') and hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'wte'):
                lm_head_ptr = model.lm_head.weight.data_ptr()
                wte_ptr = model.transformer.wte.weight.data_ptr()
                if lm_head_ptr == wte_ptr:
                    logger.info("✓ lm_head.weight правильно прив'язаний до wte")
                else:
                    logger.warning("⚠ lm_head.weight НЕ прив'язаний до wte")




        # Восстановление состояния scaler для fp16
        if fp16 and os.path.exists(emerg_scaler_state_path):
            trainer.scaler.load_state_dict(torch.load(emerg_scaler_state_path))
            logger.info("✓ Scaler state відновлено з аварійного чекпоінта")

    else:
        logger.info("Чекпоінт для відновлення тренера не знайдено або не використовується")

    logger.info(f"Модель завантажена, параметрів: {model.num_parameters():,}")
    
    # Перевірка привязки lm_head до embeddings
    #if hasattr(model, 'lm_head') and hasattr(model, 'transformer'):
    #    if hasattr(model.transformer, 'wte'):
    #        lm_head_ptr = model.lm_head.weight.data_ptr()
    #        wte_ptr = model.transformer.wte.weight.data_ptr()
    #        if lm_head_ptr == wte_ptr:
    #            logger.info("✓ lm_head.weight правильно прив'язаний до wte")
    #        else:
    #            logger.warning("⚠ lm_head.weight НЕ прив'язаний до wte")

    # Gradient Checkpointing
    if pyIanthe_config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing увімкнено")
    
    # 7. Функції тестування та метрик

    # 8. Callback для промежуточного тестування
    # 9. TrainingArguments і Trainer
    

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
                    max_new_tokens = MAX_NEW_TOKENS,
                    temperature = TEMPERATURE,
                    top_p = TOP_P,
                    do_sample = True,
                    pad_token_id = tokenizer.eos_token_id
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
            #trainer.state.save_to_json(MAIN_MODEL_DIR)
            logger.info(f"✓ Модель збережена у: {MAIN_MODEL_DIR}")
            
            # Зберігаємо аварійний чекпоінт
            emergency_checkpoint = os.path.join(CHECKPOINT_DIR, f"checkpoint-interrupted-epoch{epoch+1}")
            os.makedirs(emergency_checkpoint, exist_ok=True)
            trainer.save_model(emergency_checkpoint)
            #trainer.save_state()

            emerg_tr_state_path = os.path.join(emergency_checkpoint, "trainer_state.json")
            emerg_opt_state_path = os.path.join(emergency_checkpoint, "optimizer.pt")
            emerg_sched_state_path = os.path.join(emergency_checkpoint, "scheduler.pt")
            emerg_rng_state_path = os.path.join(emergency_checkpoint, "rng_state.pth")
            emerg_scaler_state_path = os.path.join(emergency_checkpoint, "scaler.pt")

            trainer.state.save_to_json(emerg_tr_state_path)
            if trainer.optimizer is not None:
                torch.save(trainer.optimizer.state_dict(), emerg_opt_state_path)
            if trainer.lr_scheduler is not None:
                torch.save(trainer.lr_scheduler.state_dict(), emerg_sched_state_path)

            save_all_rng_states("rngsave.txt")

            #rng_states = {
            ##    "python": random.getstate(),
             #   "numpy": np.random.get_state(),
             #   "cpu": torch.random.get_rng_state(),
           # }

            # ПРИМЕР ИСПОЛЬЗОВАНИЯ:
            # 1. Замеряем до сохранения
                #save_rng_to_file("my_rng.pt")




        


        #    if torch.cuda.is_available():

         #       cuda_states = torch.cuda.get_rng_state_all()

          #      state_bytes = cuda_states.numpy().tobytes()
           #     state_hash = hashlib.md5(state_bytes).hexdigest()
        
                # Печатаем ID устройства, размер состояния и хеш
            #    print(f"GPU: Size={cuda_states.shape[0]} bytes | Hash={state_hash}")
    



                #rng_states["cuda"] = torch.cuda.random.get_rng_state_all()

                #debug_cuda_rng_state(rng_states, logger, prefix="[SAVE]")


         #   torch.save(rng_states, emerg_rng_state_path)
         #   if fp16 and hasattr(trainer, "scaler") and trainer.scaler is not None:
         #       torch.save(trainer.scaler.state_dict(), emerg_scaler_state_path)            

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
