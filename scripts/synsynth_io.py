"""
受限文件读写：所有文件操作都通过本模块，确保路径位于 WORKSPACE_ROOT 内。
"""
from __future__ import annotations

import json
import os
from typing import Any

from synsynth_config import WORKSPACE_ROOT, safe_path, logger


def read_text(relpath: str) -> str:
    """读取工作区内的文本文件。"""
    p = safe_path(relpath)
    with open(p, encoding="utf-8") as f:
        return f.read()


def write_text(relpath: str, content: str) -> str:
    """写入文本文件，并创建所需的中间目录。"""
    p = safe_path(relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Fichier écrit → %s (%d car.)", relpath, len(content))
    return p


def write_json(relpath: str, obj: Any) -> str:
    """将对象序列化为工作区内的 JSON 文件。"""
    return write_text(relpath, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(relpath: str) -> Any:
    """读取工作区内的 JSON 文件。"""
    return json.loads(read_text(relpath))


def append_text(relpath: str, content: str) -> str:
    """向已有文件追加文本。"""
    p = safe_path(relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)
    return p
