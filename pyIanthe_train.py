# pyIanthe_train_final_callbacks.py
import os
import sys
import json
import torch
import numpy as np
import random
import math

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config,
    get_scheduler
)
from modules.pyIanthe_log import setup_logging, configure_runtime
from modules.pyIanthe_hw import get_device_config, get_num_workers
from modules.pyIanthe_utils import get_last_checkpoint, load_test_examples
from modules.pyIanthe_dataset import load_datasets
from modules.pyIanthe_metrics import compute_perplexity, compute_text_metrics
from modules.pyIanthe_test_callbacks import TestingCallback
import pyIanthe_config as cfg

if __name__ == "__main__":
    # 0. Логі та debug
    configure_runtime(cfg.DEBUG)

    logger = setup_logging(cfg.TRAINING_LOG_ENABLE, os.path.join(cfg.BASE_DIR, cfg.TRAINING_LOG_FILENAME))

    # 1. GPU / device
    device, fp16, bf16, pin_memory, num_gpus, attn_impl = get_device_config(cfg, logger)
    num_workers = get_num_workers(cfg, logger)

    # 2. Папки
    CHECKPOINT_DIR = os.path.join(cfg.BASE_DIR, cfg.FOLDER_CHECKPOINTS)
    MAIN_MODEL_DIR = os.path.join(cfg.BASE_DIR, cfg.FOLDER_MODEL)
    REPORTS_DIR = os.path.join(cfg.BASE_DIR, cfg.FOLDER_REPORTS)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MAIN_MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    LAST_CHECKPOINT = get_last_checkpoint(CHECKPOINT_DIR)

    # 3. Навчання
    EPOCHS = 3
    PER_DEVICE_BATCH_SIZE = cfg.PER_DEVICE_BATCH_SIZE
    CONTEXT_LENGTH = cfg.CONTEXT_LENGTH
    assert len(cfg.EPOCH_LRS) == EPOCHS, "Потрібно задати LR для всіх епох"
    assert len(cfg.EPOCH_SAVE_STEPS) == EPOCHS, "Потрібно задати save_steps для всіх епох"

    # 4. Датасет
    try:
        dataset, eval_dataset = load_datasets(cfg, logger)
    except Exception as e:
        logger.error(f"Помилка при завантаженні датасету: {e}")
        sys.exit(1)

    # 5. Токенізатор
    tokenizer_source_dir = os.path.join(cfg.FOLDER_MODELS, *cfg.MODEL_ID.split("/"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_dir, local_files_only=True)
    tokenizer.model_max_length = CONTEXT_LENGTH
    logger.info(f"Токенізатор завантажено з: {tokenizer_source_dir}")

    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    tokenized_dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=CONTEXT_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Модель
    vocab_size = len(tokenizer)
    main_model_exists = os.path.exists(os.path.join(MAIN_MODEL_DIR, "model.safetensors"))
    if main_model_exists:
        logger.info(f"Завантажуємо модель з {MAIN_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL_DIR,
            ignore_mismatched_sizes=False,
            local_files_only=True,
            attn_implementation=attn_impl
        ).to(device)
    else:
        logger.info("Модель не знайдена, створюємо нову")
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=CONTEXT_LENGTH,
            n_ctx=CONTEXT_LENGTH,
            n_embd=cfg.EMBEDDING_DIM,
            n_layer=cfg.NUM_LAYERS,
            n_head=cfg.HEADS,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
        )
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl).to(device)

    if cfg.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing увімкнено")

    # 7. Тестові приклади
    GENERATE_EXAMPLES = load_test_examples(logger, cfg.BASE_DIR, cfg.TEXT_TEST_FILENAME, cfg.TEXT_TESTS_COUNT)
    
    if cfg.TRAIN_OUTPUT_INFO:
        logging_strategy="no"
    else:
        logging_strategy="steps"

    # 8. Trainer базовий
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=False,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        logging_strategy=logging_strategy,
        logging_steps=cfg.TRAIN_OUTPUT_STEPS,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        report_to="none",
        disable_tqdm=False,
        save_total_limit=cfg.SAVE_LIMIT,
    )

    # Callback
    testing_callback = TestingCallback(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_prompts=GENERATE_EXAMPLES if cfg.TEST_ENABLED else None,
        eval_dataset=eval_dataset if cfg.EVAL_ENABLED else None,
        config=cfg,
        logger=logger
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset if cfg.EVAL_ENABLED else None,
        data_collator=data_collator,
        callbacks=[testing_callback]  # підключаємо callback
    )

    # 9. Обчислення steps на епоху
    effective_batch_size = PER_DEVICE_BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS * max(1, num_gpus)
    steps_per_epoch = math.ceil(len(tokenized_dataset) / effective_batch_size)
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    resume_ckpt = LAST_CHECKPOINT if LAST_CHECKPOINT else None

    # 10. Навчання по епохах з різним LR та save_steps
    for epoch_idx in range(EPOCHS):
        lr = cfg.EPOCH_LRS[epoch_idx]
        save_steps = cfg.EPOCH_SAVE_STEPS[epoch_idx]
        logger.info(f"\n===== Епоха {epoch_idx + 1} / {EPOCHS} | LR={lr} | save_steps={save_steps} =====")

        # AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)

        # Scheduler reset
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(steps_per_epoch * 0.05),
            num_training_steps=steps_per_epoch
        )

        trainer.optimizer = optimizer
        trainer.lr_scheduler = scheduler
        trainer.args.save_steps = save_steps
        trainer.args.max_steps = steps_per_epoch * (epoch_idx + 1)

        # Навчання
        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
            resume_ckpt = None
        except KeyboardInterrupt:
            logger.warning("⚠ Тренування перервано користувачем (Ctrl+C)")
            logger.info("Зберігаємо стан...")
            model.save_pretrained(MAIN_MODEL_DIR)
            tokenizer.save_pretrained(MAIN_MODEL_DIR)
            emergency_checkpoint = os.path.join(CHECKPOINT_DIR, f"checkpoint-interrupted-epoch{epoch_idx+1}")
            os.makedirs(emergency_checkpoint, exist_ok=True)
            trainer.save_model(emergency_checkpoint)
            trainer.state.save_to_json(os.path.join(emergency_checkpoint, "trainer_state.json"))
            if trainer.optimizer:
                torch.save(trainer.optimizer.state_dict(), os.path.join(emergency_checkpoint, "optimizer.pt"))
            if trainer.lr_scheduler:
                torch.save(trainer.lr_scheduler.state_dict(), os.path.join(emergency_checkpoint, "scheduler.pt"))
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state()
            }
            if torch.cuda.is_available():
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            torch.save(rng_states, os.path.join(emergency_checkpoint, "rng_state.pth"))
            logger.info(f"✓ Аварійний чекпоінт: {emergency_checkpoint}")
            sys.exit(0)

        # Зберігаємо модель після епохи
        model.save_pretrained(MAIN_MODEL_DIR)
        tokenizer.save_pretrained(MAIN_MODEL_DIR)
        checkpoint_epoch_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch_idx+1}")
        os.makedirs(checkpoint_epoch_dir, exist_ok=True)
        trainer.save_model(checkpoint_epoch_dir)
        trainer.save_state()

        # Генерація звіту по епосі
        def generate_epoch_report(epoch_index=epoch_idx):
            model.eval()
            report = {}
            for prompt in GENERATE_EXAMPLES:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=cfg.GEN_TEST_MNEW_TOKENS,
                        temperature=cfg.GEN_TEST_TEMPERATURE,
                        top_p=cfg.GEN_TEST_TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logits = model(**inputs).logits
                    ppl = compute_perplexity(logits, inputs["input_ids"])
                    avg_len, perc_meaningful = compute_text_metrics(text)
                    report[prompt] = {
                        "text": text,
                        "perplexity": ppl,
                        "avg_length": avg_len,
                        "perc_meaningful_tokens": perc_meaningful
                    }
            report_file = os.path.join(REPORTS_DIR, f"report_epoch_{epoch_index+1}.json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            model.train()
            logger.info(f"✓ Звіт по епосі {epoch_index+1} збережено: {report_file}")

        generate_epoch_report()

    logger.info("\n[SUCCESS] ТРЕНУВАННЯ ЗАВЕРШЕНО!")
    logger.info(f"Фінальна модель: {MAIN_MODEL_DIR}")
    logger.info(f"Чекпоінти: {CHECKPOINT_DIR}")
    logger.info(f"Звіти: {REPORTS_DIR}")
    logger.info(f"Лог: {cfg.TRAINING_LOG_FILENAME}")
