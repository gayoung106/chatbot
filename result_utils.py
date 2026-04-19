from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys


RESULT_DIR = Path("result")


def ensure_result_dir() -> Path:
    RESULT_DIR.mkdir(exist_ok=True)
    return RESULT_DIR


@contextmanager
def markdown_output(filename: str):
    ensure_result_dir()
    path = RESULT_DIR / filename
    handle = open(path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = handle
    try:
        yield path
    finally:
        sys.stdout = original_stdout
        handle.close()

