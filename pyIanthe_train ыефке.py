# pyIanthe_train_final.py
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, DataCollatorForLanguageModeling
from tqdm import tqdm
import math

# Імпортуємо твої модулі
from modules.pyIanthe_log import setup_logging, configure_runtime
from modules.pyIanthe_hw import get_device_config, get_num_workers
from modules.pyIanthe_dataset import load_datasets
import pyIanthe_config

# ============================================================================
# ФУНКЦІЇ ЗБЕРЕЖЕННЯ/ЗАВАНТАЖЕННЯ
# ============================================================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, step):
    """Зберігає ВСЕ в один файл"""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step
    }, path)
    print(f"✅ Збережено: epoch={epoch}, step={step}")

def load_checkpoint(path, model, optimizer, scheduler):
    """Завантажує з файлу"""
    if not os.path.exists(path):
        print("Чекпоінт не знайдено, починаємо з нуля")
        return 0, 0
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"✅ Завантажено: epoch={checkpoint['epoch']}, step={checkpoint['step']}")
    return checkpoint['epoch'], checkpoint['step']

# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # ІНІЦІАЛІЗАЦІЯ
    # ------------------------------------------------------------------------
    
    configure_runtime(pyIanthe_config.DEBUG)
    logger = setup_logging(
        log_to_file=pyIanthe_config.TRAINING_LOG_ENABLE,
        log_file=os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.TRAINING_LOG_FILENAME)
    )
    
    device, fp16, bf16, pin_memory, num_gpus, attn_impl = get_device_config(
        config=pyIanthe_config,
        logger=logger
    )
    
    num_workers = get_num_workers(config=pyIanthe_config, logger=logger)
    
    # Папки
    MODEL_DIR = os.path.join(pyIanthe_config.BASE_DIR, pyIanthe_config.FOLDER_MODEL)
    CHECKPOINT_FILE = os.path.join(pyIanthe_config.BASE_DIR, "checkpoint.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("ІНІЦІАЛІЗАЦІЯ")
    print("="*60)
    print(f"Device: {device}")
    print(f"Num workers: {num_workers}")
    
    # ------------------------------------------------------------------------
    # ЗАВАНТАЖЕННЯ ДАНИХ
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("ЗАВАНТАЖЕННЯ ДАНИХ")
    print("="*60)
    
    # Завантажуємо датасети через твій модуль
    try:
        dataset, eval_dataset = load_datasets(config=pyIanthe_config, logger=logger)
        print(f"✓ Train: {len(dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")
    except Exception as e:
        logger.error(f"❌ Помилка: {e}")
        sys.exit(1)
    
    # Токенізатор
    tokenizer_source_dir = os.path.join(
        pyIanthe_config.FOLDER_MODELS, 
        *pyIanthe_config.MODEL_ID.split("/")
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_dir, local_files_only=True)
    tokenizer.model_max_length = pyIanthe_config.CONTEXT_LENGTH
    print(f"✓ Токенізатор: vocab={len(tokenizer)}")
    
    # Фільтрація та токенізація
    dataset = dataset.filter(lambda x: isinstance(x.get("text", ""), str) and x["text"].strip())
    print(f"✓ Фільтровано: {len(dataset)} записів")
    
    tokenized_dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=pyIanthe_config.CONTEXT_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    print(f"✓ Токенізовано: {len(tokenized_dataset)} записів")
    
    # DataCollator та DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=pyIanthe_config.PER_DEVICE_BATCH_SIZE,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    print(f"✓ DataLoader: {len(dataloader)} батчів")
    
    # ------------------------------------------------------------------------
    # МОДЕЛЬ
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("МОДЕЛЬ")
    print("="*60)
    
    vocab_size = len(tokenizer)
    
    # Створюємо або завантажуємо модель
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"Завантаження з {MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            attn_implementation=attn_impl
        )
    else:
        print("Створення нової моделі")
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=pyIanthe_config.CONTEXT_LENGTH,
            n_ctx=pyIanthe_config.CONTEXT_LENGTH,
            n_embd=pyIanthe_config.EMBEDDING_DIM,
            n_layer=pyIanthe_config.NUM_LAYERS,
            n_head=pyIanthe_config.HEADS,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True
        )
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl)
    
    model = model.to(device)
    print(f"✓ Параметрів: {model.num_parameters():,}")
    
    # Gradient checkpointing тільки на GPU
    if pyIanthe_config.GRADIENT_CHECKPOINTING and device != "cpu":
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing увімкнено")
    
    # ------------------------------------------------------------------------
    # ОПТИМІЗАТОР ТА SCHEDULER
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("ОПТИМІЗАТОР")
    print("="*60)
    
    optimizer = AdamW(
        model.parameters(),
        lr=pyIanthe_config.LEARNING_RATE,
        weight_decay=pyIanthe_config.WEIGHT_DECAY
    )
    
    # Cosine scheduler з warmup
    total_steps = len(dataloader) * pyIanthe_config.EPOCHS
    warmup_steps = int(total_steps * 0.1)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"Всього кроків: {total_steps}")
    print(f"Warmup кроків: {warmup_steps}")
    print(f"LR: {pyIanthe_config.LEARNING_RATE}")
    
    # Завантажуємо чекпоінт якщо є
    start_epoch, start_step = load_checkpoint(CHECKPOINT_FILE, model, optimizer, scheduler)
    
    # ------------------------------------------------------------------------
    # ТРЕНУВАЛЬНИЙ ЦИКЛ
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("СТАРТ ТРЕНУВАННЯ")
    print("="*60)
    
    global_step = start_step
    model.train()
    
    try:
        for epoch in range(start_epoch, pyIanthe_config.EPOCHS):
            print(f"\n{'='*60}")
            print(f"ЕПОХА {epoch + 1}/{pyIanthe_config.EPOCHS}")
            print(f"{'='*60}")
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Пропускаємо батчі якщо продовжуємо з чекпоінта
                if epoch == start_epoch and batch_idx < (start_step % len(dataloader)):
                    global_step += 1
                    continue
                
                # Переносимо на device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % pyIanthe_config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Оновлюємо progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step
                })
                
                # Зберігаємо чекпоінт
                if global_step % pyIanthe_config.SAVE_STEPS == 0:
                    save_checkpoint(CHECKPOINT_FILE, model, optimizer, scheduler, epoch, global_step)
            
            # Після епохи
            print(f"\n✓ Епоха {epoch+1} завершена")
            save_checkpoint(CHECKPOINT_FILE, model, optimizer, scheduler, epoch + 1, global_step)
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("⚠ ПЕРЕРИВАННЯ (Ctrl+C)")
        print("="*60)
        save_checkpoint(CHECKPOINT_FILE, model, optimizer, scheduler, epoch, global_step)
        
        # Зберігаємо модель
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"✓ Модель збережена: {MODEL_DIR}")
        print("Для продовження - запустіть знову")
        sys.exit(0)
    
    # ------------------------------------------------------------------------
    # ЗАВЕРШЕННЯ
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("✅ ТРЕНУВАННЯ ЗАВЕРШЕНО")
    print("="*60)
    
    # Зберігаємо фінальну модель
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✓ Модель збережена: {MODEL_DIR}")
    
    # Видаляємо чекпоінт
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("✓ Чекпоінт видалено")