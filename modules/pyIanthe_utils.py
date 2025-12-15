# pyIanthe_utils.py
import os
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

