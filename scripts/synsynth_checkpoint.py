"""
SYNSYNTH+ 实验的检查点与断点续跑机制。

保存实验的中间状态，以便中断后从上次进度继续。
"""
from __future__ import annotations

import json
import os

from synsynth_config import RESULTS_DIR, logger

CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def save_checkpoint(exp_name: str, data: dict) -> None:
    """先写入临时文件，再重命名，以原子方式保存检查点。"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)  # 在 POSIX 系统上原子替换
    logger.debug("Checkpoint sauvegardé : %s (next_idx=%d)", exp_name, data.get("next_idx", -1))


def load_checkpoint(exp_name: str) -> dict | None:
    """加载已有检查点；不存在时返回 None。"""
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Checkpoint trouvé pour '%s' — reprise à idx=%d", exp_name, data.get("next_idx", 0))
        return data
    return None


def clear_checkpoint(exp_name: str) -> None:
    """实验完成后删除检查点。"""
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    if os.path.exists(path):
        os.remove(path)
        logger.info("Checkpoint supprimé : %s (expérience terminée)", exp_name)
