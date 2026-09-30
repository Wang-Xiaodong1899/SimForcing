import os

_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"


def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    _WORK_DIR = str(path) if path is not None else None
    if path is not None:
        os.makedirs(path, exist_ok=True)


def get_work_dir() -> str | None:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR
