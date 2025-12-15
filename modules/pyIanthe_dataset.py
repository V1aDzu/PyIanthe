# modules/pyIanthe_dataset.py
import os
from datasets import load_from_disk

def load_datasets(
    *,
    config,
    logger
):
    """
    Завантажує train та (опціонально) eval датасети.
    Падає з логом, якщо train датасет некоректний.
    """
    train_dataset_path = config.FOLDER_TRAIN_DATASET

    if not os.path.exists(train_dataset_path):
        logger.error(f"Папка датасету не знайдена: {train_dataset_path}")
        raise FileNotFoundError(train_dataset_path)

    arrow_files = [
        f for f in os.listdir(train_dataset_path)
        if f.endswith(".arrow")
    ]
    dataset_info = os.path.join(train_dataset_path, "dataset_info.json")

    if not arrow_files and not os.path.exists(dataset_info):
        logger.error(f"Папка {train_dataset_path} порожня або не містить датасет")
        raise RuntimeError("Invalid dataset structure")

    train_dataset = load_from_disk(train_dataset_path)
    logger.info(f"Train датасет завантажено, записів: {len(train_dataset)}")

    eval_dataset = None
    if config.EVAL_ENABLED:
        eval_path = config.FOLDER_EVAL_DATASET
        if os.path.exists(eval_path):
            try:
                eval_dataset = load_from_disk(eval_path)
                logger.info(f"Eval датасет завантажено, записів: {len(eval_dataset)}")
            except Exception as e:
                logger.warning(f"Не вдалося завантажити eval датасет: {e}")

    return train_dataset, eval_dataset
