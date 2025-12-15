# pyIanthe_debug.py
import os
import sys
import logging

def configure_runtime(debug: bool = False):
    if not debug:
        import warnings
        warnings.filterwarnings("ignore")

        import logging as std_logging
        std_logging.getLogger("transformers").setLevel(std_logging.ERROR)
        std_logging.getLogger("transformers.modeling_utils").setLevel(std_logging.ERROR)
        std_logging.getLogger("transformers.configuration_utils").setLevel(std_logging.ERROR)
        std_logging.getLogger("transformers.modeling_tf_utils").setLevel(std_logging.ERROR)

        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()

        print("[INFO] DEBUG режим вимкнено, warnings приховані")
    else:
        print("[INFO] DEBUG режим увімкнено, всі warnings показуються")

def setup_logging(log_to_file: bool = True, log_file: str | None = None, level: int = logging.INFO):
    """Налаштовує логування у файл якщо log_to_file=True та завджи у консоль"""
      
    # Створюємо logger
    logger = logging.getLogger('training')
    logger.setLevel(level)
    
    # Видаляємо старі handlers якщо є
    logger.handlers.clear()
    logger.propagate = False

    # Handler для консолі
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
  
    if log_to_file:
        if log_file is None:
            raise ValueError(
                "log_file must be provided when log_to_file=True"
            )

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(level)

    # Handler для файлу
    if log_to_file:
        if log_file is None:
            raise ValueError("log_file must be provided when log_to_file=True")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter) 
        logger.addHandler(file_handler)

    return logger
