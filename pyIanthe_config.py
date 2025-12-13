# pyIanthe.py
import os

# --- НАЛАШТУВАННЯ DEBUG ---
DEBUG = True               # True = показувати всі warnings, False = приховати
MONITOR_INTERVAL = 2

# --- HuggingFace CACHE ---
HF_HOME=r"F:\Tmp\huggingface"
HF_DATASETS_CACHE=r"F:\Tmp\huggingface\datasets"
HF_METRICS_CACHE=r"F:\Tmp\huggingface\metrics"
HF_CACHE_DIRNAME=r"F:\Tmp\huggingface\hub"
HF_HUB_WARN_DISABLE = "1"

# --- КАТАЛОГИ ---
BASE_DIR = os.path.dirname(__file__)
FOLDER_MODELS = os.path.join(BASE_DIR, "models")
FOLDER_CORPUS = "dictionaries"
FOLDER_TRAIN_DATASET = "dataset/train"
FOLDER_EVAL_DATASET = "dataset/eval"
FOLDER_CHECKPOINTS = "checkpoints"
FOLDER_REPORTS = "reports"
FOLDER_MODEL = "model"

# --- ФАЙЛИ МОДЕЛІ ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS_DATA_FILENAME = "config/datasets.json"
TEXT_TEST_FILENAME = "config/texttest.json"
TRAINING_LOG_FILENAME = "training.log"

# --- АРХІТЕКТУРА ---
EMBEDDING_DIM = 1024
NUM_LAYERS = 12
HEADS = 16

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
EPOCHS = 3                  #Епохи для основного етапу
LEARNING_RATE = 5e-4        # LR для основного етапу
CONTEXT_LENGTH = 512
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
SAVE_LIMIT = 3

# --- НАЛАШТУВАННЯ GPU ТА ПРИСКОРЕННЯ ---    
PER_DEVICE_BATCH_SIZE = 4   # базовий розмір батчу на GPU, зменшити якщо мало VRAM
NUM_WORKERS = 4             # кількість потоків на CPU
WIN_WORKERS = True
PIN_MEMORY = True           # прискорення на GPU
BF16 = True                 # Ampere/Ada GPU
FP16 = False                # половинна точність на GPU
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUMULATION_STEPS = 2  # Кількість кроків для акумуляції градієнтів
# Ефективний батч = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
# Приклади:
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=1  → Ефективний батч = 2
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=4  → Ефективний батч = 8
#   PER_DEVICE_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=2  → Ефективний батч = 8

# --- НАЛАШТУВАННЯ ТЕСТУВАННЯ ---
TEST_ENABLED = False     # Тести генерації тексту
EVAL_ENABLED = False     # Тести на eval датасеті
EVAL_STEPS = 110         # Кожні 500 кроків
TEXT_TESTS_COUNT = 10    # 10 промптів
EVAL_TESTS_COUNT = 10    # 10 примерів из eval

# --- НАЛАШТУВАННЯ ПОРІВНЯННЯ ---
EVAL_PERCENT = 5

#        Recommended eval sizes 
#  > 1M зап   1-2%   980K-990K   10K-20K
# 100K - 1M     5%    95K-950K    5K-50K
# 10K - 100K   10%      9K-90K    1K-10K
# 1K - 10K  15-20%    800-8.5K    200-2K
#  < 1K зап 20-25%        <800   200-250