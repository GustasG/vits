import logging
import os
import sys
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def get_logger(model_dir: Path, filename: str):
    global logger

    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    model_dir.mkdir(parents=True, exist_ok=True)

    h = logging.FileHandler(model_dir / filename)
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)

    return logger
