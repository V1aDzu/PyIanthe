import os
import shutil
import glob
import re
import datetime
import numpy as np

# Убедитесь, что все необходимые библиотеки установлены:
# pip install transformers datasets accelerate torch scikit-learn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    set_seed
)
from datasets import load_dataset, IterableDataset, load_metric
from torch.optim import AdamW

# --- ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ---

# Основные настройки
MODEL_NAME = "bert-base-uncased"  # Ваша предобученная модель
TASK_NUM_LABELS = 2               # Количество классов
SEED = 42
set_seed(SEED)

# Настройки путей
CHECKPOINTS_DIR = "checkpoints"    # Основная папка для чекпоинтов Trainer
MODEL_SAVE_DIR = "final_model"     # Папка для сохранения финальной модели/весов
SAFE_ARCHIVE_DIR = "safe_archives" # Папка для надежных резервных копий (веса + токенизатор)
LOCAL_DATA_DIR = "Dataset"         # Корневая папка для локальных данных

# Настройки обучения
EVAL_STEPS = 500                   # Как часто проводить оценку (шаги)
SAVE_STEPS = 500                   # Как часто сохранять чекпоинт (шаги)
MAX_STEPS = 10000                  # Максимальное количество шагов 
TRAIN_EPOCHS = 1                   # Количество эпох (используйте 1 для IterableDataset)

# Гиперпараметры
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# --- ПОЛЬЗОВАТЕЛЬСКИЕ CALLBACK-И ---

class ReliableBackupCallback(TrainerCallback):
    """
    Создает надежную резервную копию только весов и токенизатора 
    с пользовательским именованием в отдельной папке SAFE_ARCHIVE_DIR.
    """
    def __init__(self, safe_archive_dir):
        self.safe_archive_dir = safe_archive_dir
        os.makedirs(self.safe_archive_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        """Вызывается после того, как Trainer сохранил свой стандартный чекпоинт."""
        
        # 1. Формирование имени папки
        now = datetime.datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%Y")
        backup_name = f"checkpoint_{timestamp}_{datestamp}"
        backup_path = os.path.join(self.safe_archive_dir, backup_name)
        
        # 2. Сохраняем ТОЛЬКО веса и токенизатор
        model_to_save = kwargs['model']
        tokenizer = kwargs['tokenizer']
        
        # Сохранение весов
        model_to_save.save_pretrained(backup_path)
        # Сохранение токенизатора
        if tokenizer is not None:
            tokenizer.save_pretrained(backup_path)
            
        print(f"\n[РЕЗЕРВ] Создана надежная копия весов: {backup_path}")

    def on_train_end(self, args, state, control, **kwargs):
        """Сохранение финальной модели в отдельную папку после завершения обучения."""
        model_to_save = kwargs['model']
        tokenizer = kwargs['tokenizer']
        
        # Сохранение финальной модели
        model_to_save.save_pretrained(MODEL_SAVE_DIR)
        if tokenizer is not None:
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print(f"\n[ФИНАЛ] Финальная модель сохранена в: {MODEL_SAVE_DIR}")

class CustomCheckpointingCallback(TrainerCallback):
    """
    Переименование стандартного чекпоинта Trainer сразу после его сохранения
    в формат checkpoint_<step>_<HHMMSS>_<DDMMYYYY>.
    """
    def on_save(self, args, state, control, **kwargs):
        """Вызывается после того, как Trainer сохранил свой стандартный чекпоинт."""
        
        # Поиск последнего сохраненного чекпоинта (например, "checkpoints/checkpoint-500")
        list_of_files = glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        if not list_of_files:
            return

        # Ищем самый последний файл по времени создания
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        
        # Формирование нового имени
        now = datetime.datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%Y")
        
        # Извлечение номера шага (step) из оригинального имени
        step_match = re.search(r'checkpoint-(\d+)', latest_checkpoint)
        step_number = step_match.group(1) if step_match else "unknown"
        
        new_name = f"checkpoint_{step_number}_{timestamp}_{datestamp}"
        new_path = os.path.join(args.output_dir, new_name)
        
        # Переименование
        shutil.move(latest_checkpoint, new_path)
        print(f"\n[CHECKPOINT] Переименован в: {new_path}")

# --- ФУНКЦИИ ПОДГОТОВКИ ---

def compute_metrics(p):
    """Метрика точности для бинарной классификации."""
    preds = np.argmax(p.predictions, axis=1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=preds, references=p.label_ids)

def find_latest_checkpoint(path):
    """Ищет самую последнюю (по времени создания) папку чекпоинта."""
    list_of_dirs = glob.glob(os.path.join(path, 'checkpoint_*'))
    if not list_of_dirs:
        return None
    # Сортировка по времени модификации
    latest_checkpoint = max(list_of_dirs, key=os.path.getmtime)
    return latest_checkpoint

def load_and_tokenize_data(tokenizer):
    """
    Загружает локальные данные в формате IterableDataset.
    Вам нужно, чтобы в папке {LOCAL_DATA_DIR} находились train.jsonl и eval.jsonl
    """
    print(f"\n[DATA] Загрузка данных из {LOCAL_DATA_DIR}/...")
    
    # 1. Загрузка данных (предполагаем формат JSON Lines)
    try:
        raw_datasets = load_dataset(
            'json', 
            data_files={
                'train': os.path.join(LOCAL_DATA_DIR, 'train.jsonl'), 
                'eval': os.path.join(LOCAL_DATA_DIR, 'eval.jsonl')
            }, 
            streaming=True # Использование IterableDataset (потоковая загрузка)
        )
        train_ds = raw_datasets['train']
        eval_ds = raw_datasets['eval']
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить локальные файлы: {e}")
        print("Использование примера публичного датасета 'imdb' для демонстрации.")
        # Запасной вариант для тестирования:
        raw_datasets = load_dataset("imdb", split=['train', 'test'], streaming=True)
        train_ds = raw_datasets[0]
        eval_ds = raw_datasets[1]


    def tokenize_function(examples):
        # Обязательно убедитесь, что ваш текстовый столбец называется 'text' или измените здесь
        return tokenizer(examples["text"], truncation=True)

    # Применение токенизации к IterableDataset 
    # (может быть медленным, но это необходимость для потоковой загрузки)
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_eval_ds = eval_ds.map(tokenize_function, batched=True)
    
    return tokenized_train_ds, tokenized_eval_ds

# --- ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ---

def run_training():
    # 0. Создание необходимых папок
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SAFE_ARCHIVE_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # --- 1. Поиск Чекпоинтов для Восстановления ---
    
    # 1.1 Поиск последнего полного чекпоинта Trainer (для восстановления состояния оптимизатора)
    full_checkpoint_path = find_latest_checkpoint(CHECKPOINTS_DIR)

    # 1.2 Поиск последней резервной копии весов (на случай ошибки Trainer)
    latest_archive_path = find_latest_checkpoint(SAFE_ARCHIVE_DIR)

    # --- 2. Инициализация Модели и Токенизатора ---

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    resume_arg = None
    
    if full_checkpoint_path:
        # ПРИОРИТЕТ 1: Старт с полного рабочего чекпоинта Trainer
        print(f"\n[ЗАПУСК] Возобновление обучения из полного чекпоинта: {full_checkpoint_path}")
        model = AutoModelForSequenceClassification.from_pretrained(full_checkpoint_path, num_labels=TASK_NUM_LABELS)
        resume_arg = full_checkpoint_path # Передаем путь для восстановления состояния Trainer
        
    elif latest_archive_path:
        # ПРИОРИТЕТ 2: Резервный старт с весов из архива (теряем состояние оптимизатора, но спасаем веса)
        print(f"\n[ЗАПУСК] Резервный старт с весов из архива: {latest_archive_path}")
        model = AutoModelForSequenceClassification.from_pretrained(latest_archive_path, num_labels=TASK_NUM_LABELS)
        # resume_arg остается None, чтобы Trainer начал обучение как новое, но с уже загруженными весами
        
    else:
        # ПРИОРИТЕТ 3: Старт с нуля
        print("\n[ЗАПУСК] Старт обучения с нуля.")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=TASK_NUM_LABELS)

    # --- 3. Настройка Тренера и Аргументов ---
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,             
        num_train_epochs=TRAIN_EPOCHS,
        max_steps=MAX_STEPS,                    
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        
        # Стратегия сохранения
        evaluation_strategy="steps",             
        eval_steps=EVAL_STEPS,
        save_strategy="steps",                   
        save_steps=SAVE_STEPS,
        save_total_limit=3,                      # Хранить 3 последних полных чекпоинта
        load_best_model_at_end=True,            
        metric_for_best_model="accuracy",
        
        # Настройки LR и Шедулера: Linear Warmup -> Cosine Decay
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=100,
        lr_scheduler_type="cosine",             # Cosine Decay
        warmup_steps=int(MAX_STEPS * 0.1),      # 10% шагов на Warmup
        
        # Настройки для CPU/общих:
        no_cuda=True,                           # Принудительно используем CPU (если не нужно GPU)
    )

    # --- 4. Загрузка Данных, Коллатор ---
    train_ds, eval_ds = load_and_tokenize_data(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5. Инициализация Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        
        # Добавление Callback-ов
        callbacks=[
            ReliableBackupCallback(safe_archive_dir=SAFE_ARCHIVE_DIR), 
            CustomCheckpointingCallback(),                            
        ]
    )
    
    # --- 6. Запуск Обучения ---
    try:
        trainer.train(resume_from_checkpoint=resume_arg)
    except KeyboardInterrupt:
        # Эта логика срабатывает, если Trainer не успел обработать Ctrl+C сам.
        # В большинстве случаев Trainer сам сохраняет аварийный чекпоинт, 
        # но если прерывание жесткое, то сработает логика загрузки из архива при следующем запуске.
        print("\n[ПРЕРЫВАНИЕ] Обучение прервано пользователем (Ctrl+C). Поиск последнего рабочего чекпоинта при следующем запуске.")


if __name__ == "__main__":
    run_training()