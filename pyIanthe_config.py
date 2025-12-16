# pyIanthe_config.py
import os

# --- НАЛАШТУВАННЯ DEBUG ---
DEBUG = True                      # True = показувати всі warnings, False = приховати
TRAINING_LOG_ENABLE = False
TRAINING_LOG_FILENAME = "training.log"
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

# --- АРХІТЕКТУРА ---
EMBEDDING_DIM = 896
NUM_LAYERS = 10
HEADS = 14

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
EPOCHS = 2                          # кількість епох тренування
EPOCH_LRS = [4e-4, 2e-4, 1e-4]
EPOCH_SAVE_STEPS = [5, 250, 500]    # кількість шагів між збереженням чекпоінтів
TRAIN_OUTPUT_INFO = True        # відображення повідомлень щодо тренування
TRAIN_OUTPUT_STEPS = 3          # як часто будуть відображатись повідомлення щодо тренування

# --- НАЛАШТУВАННЯ ЗБЕРЕЖЕННЯ ---
SAVE_STEPS = 5
SAVE_LIMIT = 3

# --- НАЛАШТУВАННЯ ТРЕНУВАННЯ ---
LEARNING_RATE = 4e-4            # LR для основного етапу
CONTEXT_LENGTH = 512
WEIGHT_DECAY = 0.01
GRADIENT_CLIPPING = 1.0

# --- НАЛАШТУВАННЯ GPU ТА ПРИСКОРЕННЯ ---    
PER_DEVICE_BATCH_SIZE = 8       # базовий розмір батчу на GPU, зменшити якщо мало VRAM
NUM_WORKERS = 0                 # кількість потоків на CPU
WIN_WORKERS = False
PIN_MEMORY = True               # прискорення на GPU
BF16 = False                    # Ampere/Ada GPU (вимкнено для CPU)
FP16 = False                    # половинна точність на GPU (вимкнено для CPU)
ATTENTION_TYPE = "eager"        # "eager"- звичайний (без),"sdpa" - флеш атт, "flash_attention_2" - флеш атт 2
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_STEPS = 1 # Кількість кроків для акумуляції градієнтів

# --- НАЛАШТУВАННЯ ТЕСТУВАННЯ ---
EVAL_PERCENT = 5                # load_datasets take EVAL_PERCENT% for eval dataset
TEST_ENABLED = True             # Тести генерації тексту
EVAL_ENABLED = True             # Тести на eval датасеті
EVAL_STEPS = 10                 # Кожні 10 кроків
TEXT_TESTS_COUNT = 9            # 9 промптів
EVAL_TESTS_COUNT = 9            # 9 примерів из eval
GEN_TEST_MNEW_TOKENS = 50
GEN_TEST_TEMPERATURE = 0.8
GEN_TEST_TOP_P = 0.9
