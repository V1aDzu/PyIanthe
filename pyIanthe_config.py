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
TF_CACHE_DIRNAME=r"F:\Tmp\huggingface\TF"
HF_HUB_WARN_DISABLE = "ON"

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
EMBEDDING_DIM = 894
NUM_LAYERS = 10
HEADS = 14

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
EPOCHS = 3                  #Епохи для основного етапу
LEARNING_RATE = 5e-4        # LR для основного етапу
CONTEXT_LENGTH = 512
WEIGHT_DECAY = 0.01
SAVE_STEPS = 5000
SAVE_LIMIT = 3

# --- НАЛАШТУВАННЯ GPU ТА ПРИСКОРЕННЯ ---    
PER_DEVICE_BATCH_SIZE = 6   # базовий розмір батчу на GPU, зменшити якщо мало VRAM
NUM_WORKERS = 6             # кількість потоків на CPU
WIN_WORKERS = True
PIN_MEMORY = True           # прискорення на GPU
BF16 = True                 # Ampere/Ada GPU
FP16 = False                # половинна точність на GPU
ATTENTION_TYPE = "sdpa"    # "eager"  - звичайний, без
                            #  "sdpa"   - флеш аттеншен,
                            #  "flash_attention_2" - флеш аттеншен 2, потрібні налаштування
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUMULATION_STEPS = 2  # Кількість кроків для акумуляції градієнтів
# Ефективний батч = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
# Приклади:
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=1  → Ефективний батч = 2
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=4  → Ефективний батч = 8
#   PER_DEVICE_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=2  → Ефективний батч = 8

# --- НАЛАШТУВАННЯ ТЕСТУВАННЯ ---
TEST_ENABLED = True     # Тести генерації тексту
EVAL_ENABLED = True     # Тести на eval датасеті
EVAL_STEPS = 3000       # Кожні 500 кроків
TEXT_TESTS_COUNT = 9    # 10 промптів
EVAL_TESTS_COUNT = 9    # 10 примерів из eval

# --- НАЛАШТУВАННЯ ПОРІВНЯННЯ ---
EVAL_PERCENT = 5

#        Recommended eval sizes 
#  > 1M зап   1-2%   980K-990K   10K-20K
# 100K - 1M     5%    95K-950K    5K-50K
# 10K - 100K   10%      9K-90K    1K-10K
# 1K - 10K  15-20%    800-8.5K    200-2K
#  < 1K зап 20-25%        <800   200-250