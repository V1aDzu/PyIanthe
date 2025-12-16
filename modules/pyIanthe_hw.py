# modules/pyIanthe_hw.py
import platform
import logging
import torch

def get_device_config(config=None, logger=None):
    """
    Повертає конфігурацію пристрою для тренування.
    
    Args:
        config: конфігураційний об'єкт з атрибутами FP16, BF16, PIN_MEMORY, ATTENTION_TYPE.
                Якщо None, використовуються дефолтні значення.
        logger: Логер для виведення інформації. Якщо немає, вивід через print.
    
    Returns: device, fp16, bf16, pin_memory, num_gpus, attn_impl
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    fp16 = config.FP16 if device == "cuda" else False
    bf16 = config.BF16 if device == "cuda" else False
    pin_memory = config.PIN_MEMORY if device == "cuda" else False
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Attention type
    attn_impl = config.ATTENTION_TYPE if device == "cuda" else "eager"
    logger.info(f"Attention implementation: {attn_impl}")

    logger.info(f"Пристрій: {device}, GPU: {num_gpus}")
    if device == "cuda":
        logger.info(f"GPU знайдено: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Тренування буде на CPU")

    return device, fp16, bf16, pin_memory, num_gpus, attn_impl


def get_num_workers(    
    config,
    logger: logging.Logger
) -> int:
    """
    Визначає фактичну кількість воркерів залежно від платформи та налаштувань
    
    Логіка:
    - Linux/Mac → завжди використовувати NUM_WORKERS з конфігу
    - Windows + WIN_WORKERS=True → використовувати NUM_WORKERS з конфігу
    - Windows + WIN_WORKERS=False → примусово 0 (безпечний режим)
    """
    system = platform.system()
    config_workers = config.NUM_WORKERS
    
    if system == "Windows":
        if config.WIN_WORKERS:
            logger.info(f"[NUM_WORKERS] Windows виявлено, WIN_WORKERS=True")
            logger.info(f"[NUM_WORKERS] Використовується NUM_WORKERS={config_workers}")
            logger.warning(f"[NUM_WORKERS] ⚠ Експериментальний режим! Можливі проблеми зі стабільністю")
            return config_workers
        else:
            logger.info(f"[NUM_WORKERS] Windows виявлено, WIN_WORKERS=False")
            logger.info(f"[NUM_WORKERS] Примусово встановлено NUM_WORKERS=0 (безпечний режим)")
            if config_workers > 0:
                logger.info(f"[NUM_WORKERS] NUM_WORKERS={config_workers} з конфігу проігноровано")
            return 0
    else:
        logger.info(f"[NUM_WORKERS] {system} виявлено")
        logger.info(f"[NUM_WORKERS] Використовується NUM_WORKERS={config_workers}")
        return config_workers
