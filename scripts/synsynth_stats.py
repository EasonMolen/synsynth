"""
SYNSYNTH+ 统计工具：Bootstrap 置信区间及其他指标。
"""
from __future__ import annotations

import random
from typing import Sequence

from synsynth_config import RANDOM_SEED


def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> dict:
    """使用 Bootstrap 计算均值及置信区间。

    返回：
        {"mean": float, "ci_low": float, "ci_high": float, "std": float}
    """
    if not scores:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    mean_val = sum(scores) / n
    std_val = (sum((x - mean_val) ** 2 for x in scores) / n) ** 0.5

    return {
        "mean": round(mean_val, 4),
        "ci_low": round(means[lo_idx], 4),
        "ci_high": round(means[min(hi_idx, n_bootstrap - 1)], 4),
        "std": round(std_val, 4),
    }


def token_f1(pred: str, gold: str) -> float:
    """计算词元级 F1（HotpotQA 标准指标）。"""
    pred_tokens = _normalize_tokens(pred)
    gold_tokens = _normalize_tokens(gold)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_tokens(text: str) -> list[str]:
    """分词并归一化，以计算 F1。"""
    import re
    text = text.lower().strip()
    # 去除英语和法语中的常见冠词
    text = re.sub(r"\b(the|a|an|le|la|les|un|une|des|l'|d')\b", " ", text)
    tokens = text.split()
    return [t for t in tokens if t]
