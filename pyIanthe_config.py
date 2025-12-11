# pyIanthe.py
import os

# --- НАЛАШТУВАННЯ DEBUG ---
DEBUG = False               # True = показувати всі warnings, False = приховати
MONITOR_INTERVAL = 2

# --- КАТАЛОГИ ---
BASE_DIR = os.path.dirname(__file__)
FOLDER_MODELS = os.path.join(BASE_DIR, "models")
FOLDER_CORPUS = "dictionaries"
FOLDER_TRAIN_DATASET = "dataset"
FOLDER_CHECKPOINTS = "checkpoints"
FOLDER_REPORTS = "reports"
FOLDER_MODEL = "model"
#FOLDER_TOKENIZER = "tokenizer"

# --- ФАЙЛИ МОДЕЛІ ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS_DATA_FILENAME = "datasets.json"

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
PER_DEVICE_BATCH_SIZE = 2   # зменшити якщо мало VRAM
NUM_WORKERS = 2             # кількість потоків на CPU
PIN_MEMORY = True           # прискорення на GPU
FP16 = True                 # половинна точність на GPU
