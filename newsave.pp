Функция ПОЛНОГО аварийного сохранения (КЛЮЧЕВОЕ)
Что сохраняем в model/
model/
├── model.safetensors
├── config.json
├── tokenizer/
├── training_state.json        ← НАШ файл
├── optimizer.pt               ← опционально
├── scheduler.pt               ← опционально
├── scaler.pt                  ← если fp16
├── rng_state.pth

Пример функции
def emergency_full_save(model, tokenizer, trainer, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Модель + токенизатор
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 2. RNG
    torch.save({
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, os.path.join(output_dir, "rng_state.pth"))

    # 3. Optimizer / Scheduler / Scaler (если есть)
    if trainer.optimizer:
        torch.save(trainer.optimizer.state_dict(),
                   os.path.join(output_dir, "optimizer.pt"))

    if trainer.lr_scheduler:
        torch.save(trainer.lr_scheduler.state_dict(),
                   os.path.join(output_dir, "scheduler.pt"))

    if hasattr(trainer, "scaler") and trainer.scaler:
        torch.save(trainer.scaler.state_dict(),
                   os.path.join(output_dir, "scaler.pt"))

    # 4. НАШ state (не HF)
    state = {
        "global_step": trainer.state.global_step,
        "epoch": float(trainer.state.epoch or 0),
        "saved_at": datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


❗ Это не HF checkpoint. Это твой атомарный snapshot.

2️⃣ Как ВОССТАНАВЛИВАТЬСЯ (гибко!)
def load_soft_resume(model_dir, model, trainer):
    state_file = os.path.join(model_dir, "training_state.json")
    if not os.path.exists(state_file):
        return 0

    with open(state_file) as f:
        state = json.load(f)

    # optimizer — ТОЛЬКО если ты хочешь
    opt_path = os.path.join(model_dir, "optimizer.pt")
    if os.path.exists(opt_path) and trainer.optimizer:
        trainer.optimizer.load_state_dict(torch.load(opt_path))

    # scheduler — ТОЛЬКО если совместим
    sch_path = os.path.join(model_dir, "scheduler.pt")
    if os.path.exists(sch_path) and trainer.lr_scheduler:
        trainer.lr_scheduler.load_state_dict(torch.load(sch_path))

    # scaler
    scaler_path = os.path.join(model_dir, "scaler.pt")
    if os.path.exists(scaler_path) and hasattr(trainer, "scaler"):
        trainer.scaler.load_state_dict(torch.load(scaler_path))

    return state["global_step"]

    Отключаем HF-чекпоинты как механизм восстановления
❌ УБРАТЬ полностью
last_checkpoint = get_last_checkpoint()
resume_from = last_checkpoint


И всю логику:

resume_from_checkpoint=resume_from


HF checkpoint больше не используется для resume.

2️⃣ Упрощаем загрузку модели (ключевое место)
🔁 ЗАМЕНИТЬ БЛОК ЗАГРУЗКИ МОДЕЛИ НА ЭТО
MODEL_STATE_DIR = MAIN_MODEL_DIR

if os.path.exists(os.path.join(MODEL_STATE_DIR, "model.safetensors")):
    logger.info(f"Завантажуємо модель з: {MODEL_STATE_DIR}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_STATE_DIR,
        local_files_only=True,
        attn_implementation=attn_impl
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_STATE_DIR,
        local_files_only=True
    )

else:
    logger.info("Модель не знайдена — створюємо нову")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
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


❗ НИКАКИХ resume_from_checkpoint

3️⃣ TrainingArguments — только из текущего конфига

Это уже почти правильно, но надо убрать зависимость от старого state.

❗ ИЗМЕНИТЬ
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    overwrite_output_dir=True,  # важно
    num_train_epochs=1,
    save_steps=SAVE_STEPS,
    save_strategy="steps",
    save_total_limit=pyIanthe_config.SAVE_LIMIT,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    ...
)


HF может сохранять, но мы НЕ используем это для восстановления.

4️⃣ Главная функция EMERGENCY SAVE (ключевая часть)
🔥 ДОБАВЬ ЭТУ ФУНКЦИЮ
def emergency_full_save(model, tokenizer, trainer, target_dir):
    logger.warning("⚠ EMERGENCY SAVE: зберігаємо ПОВНИЙ стан")

    tmp_dir = target_dir + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    torch.save(trainer.optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pt"))
    torch.save(trainer.lr_scheduler.state_dict(), os.path.join(tmp_dir, "scheduler.pt"))

    with open(os.path.join(tmp_dir, "trainer_meta.json"), "w") as f:
        json.dump({
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        }, f, indent=2)

    # атомарная замена
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.rename(tmp_dir, target_dir)

    logger.warning(f"✓ Emergency save complete → {target_dir}")

5️⃣ KeyboardInterrupt — ПРАВИЛЬНО
🔁 ЗАМЕНИ ВЕСЬ except KeyboardInterrupt НА:
except KeyboardInterrupt:
    logger.error("⚡ АВАРІЙНЕ ПЕРЕРИВАННЯ (немає світла / Ctrl+C)")

    emergency_full_save(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        target_dir=MAIN_MODEL_DIR
    )

    sys.exit(0)


❗ НИКАКИХ trainer.save_state()
❗ НИКАКИХ копирований из CHECKPOINT_DIR

6️⃣ Восстановление после перезапуска (как это работает)

Свет пропал → emergency save

Свет появился → ты:

меняешь LEARNING_RATE

меняешь GRADIENT_ACCUMULATION_STEPS

меняешь что угодно

Запуск скрипта:

модель загружается

optimizer создаётся заново

scheduler создаётся заново

обучение продолжается стабильно

Да, LR «скакнёт» — и это нормально, ты сам этого хочешь.