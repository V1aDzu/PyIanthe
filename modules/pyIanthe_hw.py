import platform
import logging

def get_num_workers(
    *,
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
