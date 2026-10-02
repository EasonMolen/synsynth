"""
SYNSYNTH+ 的答案忠实度检查器。

先将回答拆分成独立主张，再用自然语言推断对照原始上下文验证每条主张。
从两个角度使用大语言模型评审，以减小结果方差。
"""
from __future__ import annotations

import json
import re
from typing import Any

from synsynth_config import logger
from synsynth_model import generate_structured


# ── 提示词 ─────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = (
    "Décompose la réponse suivante en affirmations atomiques (claims). "
    "Chaque claim doit être une phrase simple et indépendante. "
    "Réponds UNIQUEMENT en JSON : {\"claims\": [\"claim 1\", \"claim 2\", ...]}"
)

_NLI_PROMPT = (
    "Tu es un vérificateur de faits. Pour chaque affirmation, détermine si "
    "elle est SUPPORTÉE par le contexte (le contexte contient l'information, "
    "même reformulée ou paraphrasée) ou NON-SUPPORTÉE (information inventée "
    "ou absente du contexte). "
    "Réponds UNIQUEMENT en JSON : "
    "{\"verdicts\": [{\"claim\": \"...\", \"supported\": true/false}, ...]}"
)


def decompose_claims(answer: str) -> list[str]:
    """使用大语言模型将回答拆分为独立主张。"""
    prompt = f"Réponse à décomposer :\n{answer}"
    raw = generate_structured(prompt, system=_DECOMPOSE_PROMPT, max_new_tokens=512)

    # 提取 JSON
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            claims = obj.get("claims", [])
            if isinstance(claims, list) and claims:
                return [str(c) for c in claims]
    except (json.JSONDecodeError, ValueError):
        pass

    # 失败时按句子拆分
    sentences = re.split(r"[.!?]+", answer)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def verify_claims(claims: list[str], context: str) -> list[dict]:
    """通过自然语言推断，逐条核对主张与上下文。"""
    if not claims:
        return []

    claims_text = "\n".join(f"- {c}" for c in claims)
    prompt = (
        f"Contexte :\n{context}\n\n"
        f"Affirmations à vérifier :\n{claims_text}"
    )
    raw = generate_structured(prompt, system=_NLI_PROMPT, max_new_tokens=1024)

    # 提取验证结论
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            verdicts = obj.get("verdicts", [])
            if isinstance(verdicts, list):
                return verdicts
    except (json.JSONDecodeError, ValueError):
        pass

    # 失败时视为无法验证
    return [{"claim": c, "supported": False} for c in claims]


def compute_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """完整流程：拆分主张、验证、计算得分。

    返回：
        {"faithfulness": float, "total_claims": int, "supported": int,
         "claims": [{"claim": str, "supported": bool}, ...]}
    """
    claims = decompose_claims(answer)
    if not claims:
        return {"faithfulness": 1.0, "total_claims": 0, "supported": 0, "claims": []}

    verdicts = verify_claims(claims, context)

    # 将验证结论与各条主张对应
    supported = sum(1 for v in verdicts if v.get("supported", False))
    total = len(claims)
    score = supported / total if total > 0 else 0.0

    return {
        "faithfulness": round(score, 4),
        "total_claims": total,
        "supported": supported,
        "claims": verdicts,
    }
