# pyIanthe_utils.py
import os
import json
from typing import Optional

def get_last_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Повертає шлях до останнього чекпоінта або None, якщо чекпоінтів немає
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not dirs:
        return None

    # Пріоритет: checkpoint-N
    checkpoint_dirs = []
    for d in dirs:
        if d.startswith("checkpoint-"):
            try:
                checkpoint_dirs.append((int(d.split("-")[-1]), d))
            except ValueError:
                pass

    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: x[0])
        return os.path.join(checkpoint_dir, checkpoint_dirs[-1][1])

    # fallback: лексикографічно
    dirs.sort()
    return os.path.join(checkpoint_dir, dirs[-1])

def load_test_examples(logger, base_dir, filename, max_examples=15):
    """
    Завантажує тестові приклади з JSON файлу.
    
    Параметри:
    - logger: об'єкт логування
    - base_dir: директорія, де шукати файл
    - filename: ім'я файлу з прикладами
    - max_examples: максимальна кількість прикладів для повернення
    
    Повертає список рядків (тестові приклади).
    Якщо файл не знайдено або невірний формат — повертає стандартні приклади.
    """
    test_file = os.path.join(base_dir, filename)
    default_examples = ["Привіт, як справи?", "Одного разу", "Швидка коричнева лисиця"]
    
    if not os.path.exists(test_file):
        logger.warning(f"Файл тестових прикладів не знайдено: {test_file}")
        logger.info("Використовуються стандартні приклади")
        return default_examples
    
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            examples = data[:max_examples]
            logger.info(f"Завантажено {len(examples)} тестових прикладів з {test_file}")
            return examples
        
        elif isinstance(data, dict) and "examples" in data:
            examples = data["examples"][:max_examples]
            logger.info(f"Завантажено {len(examples)} тестових прикладів з {test_file}")
            return examples
        
        else:
            logger.warning(f"Невідомий формат файлу {test_file}")
            logger.info("Використовуються стандартні приклади")
            return default_examples
        
    except Exception as e:
        logger.error(f"Помилка при завантаженні {test_file}: {e}")
        logger.info("Використовуються стандартні приклади")
        return default_examples
