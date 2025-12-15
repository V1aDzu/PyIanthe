# pyIanthe.py
import os

# --- НАЛАШТУВАННЯ DEBUG ---
DEBUG = True              # True = показувати всі warnings, False = приховати
TRAINING_LOG_ENABLE = True
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
EMBEDDING_DIM = 896
NUM_LAYERS = 10
HEADS = 14

# --- РЕЖИМ ТРЕНУВАННЯ ---
TRAINING_MODE = "epochs"        # "steps" | "epochs"
MAX_STEPS = -1                  # якщо steps -1 = без обмеження
EPOCHS = 2                      # якщо epochs
TRAIN_OUTPUT_INFO = True
TRAIN_OUTPUT_STEPS = 3

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
SAVE_STEPS = 5
SAVE_LIMIT = 3

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
LEARNING_RATE = 4e-4            #5e-4        # LR для основного етапу
CONTEXT_LENGTH = 512
WEIGHT_DECAY = 0.01
GRADIENT_CLIPPING=1.0

# --- НАЛАШТУВАННЯ GPU ТА ПРИСКОРЕННЯ ---    
PER_DEVICE_BATCH_SIZE = 8       # базовий розмір батчу на GPU, зменшити якщо мало VRAM
NUM_WORKERS = 4                 # кількість потоків на CPU
WIN_WORKERS = True
PIN_MEMORY = True               # прискорення на GPU
BF16 = True                     # Ampere/Ada GPU
FP16 = False                    # половинна точність на GPU
ATTENTION_TYPE = "spda"         # "eager"- звичайний (без),"sdpa" - флеш атт, "flash_attention_2" - флеш атт 2
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_STEPS = 1 # Кількість кроків для акумуляції градієнтів

# --- НАЛАШТУВАННЯ ТЕСТУВАННЯ ---
EVAL_PERCENT = 5                # load_datasets take EVAL_PERCENT% for eval dataset
TEST_ENABLED = True             # Тести генерації тексту
EVAL_ENABLED = True             # Тести на eval датасеті
EVAL_STEPS = 10                 # Кожні 500 кроків
TEXT_TESTS_COUNT = 9            # 10 промптів
EVAL_TESTS_COUNT = 9            # 10 примерів из eval
GEN_TEST_MNEW_TOKENS = 50
GEN_TEST_TEMPERATURE = 0.8
GEN_TEST_TOP_P = 0.9



# Effective batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
# Examples:
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=1  → Effective batch = 2
#   PER_DEVICE_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=4  → Effective batch = 8
#   PER_DEVICE_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=2  → Effective batch = 8

# Learning rate
#│       ╭───────╮
#│      ╱         ╲
#│     ╱           ╲
#│____╱             ╲____
#     warmup      decay
#Size    Safe start LR       Work    Comment
#0.5B        5e-4    	3e-4 – 1e-3	Can be agressive
#1.5B	    3e-4	    2e-4 – 5e-4	Optimal
#  3B	    2e-4	    1e-4 – 3e-4	Sensitive
#  7B	    1e-4	    5e-5 – 2e-4	Standart
# 13B	    7e-5	    3e-5 – 1e-4	Carefully
# 30B	    3e-5	    1e-5 – 5e-5	Fragile

#        Recommended eval sizes 
#  > 1M зап   1-2%   980K-990K   10K-20K
# 100K - 1M     5%    95K-950K    5K-50K
# 10K - 100K   10%      9K-90K    1K-10K
# 1K - 10K  15-20%    800-8.5K    200-2K
#  < 1K зап 20-25%        <800   200-250