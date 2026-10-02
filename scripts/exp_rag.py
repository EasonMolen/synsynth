"""
实验 4：RAG 与忠实度评估（RAGAS 风格）。

三个 RAGAS 指标：
  1. Faithfulness：答案是否忠于给定上下文？
  2. Answer Relevance：答案是否切题？
  3. Context Precision：上下文是否包含所需信息？

使用 Gemma-4 作为评审模型。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from synsynth_config import logger, TARGET_FAITHFULNESS
from synsynth_model import generate, generate_structured
from synsynth_data import load_rag_data
from synsynth_stats import bootstrap_ci
from faithfulness_checker import compute_faithfulness
from synsynth_checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

# ── 评审提示词 ─────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "Tu es un assistant de réponse aux questions basé sur un graphe de "
    "connaissances (KG-RAG). Tu ne dois utiliser QUE le contexte fourni. "
    "Si l'information n'est pas dans le contexte, réponds 'Information non disponible'.\n\n"
    "Exemple :\n"
    "Contexte : Paris est la capitale de la France depuis 508.\n"
    "Question : Quelle est la capitale de la France ?\n"
    "Réponse : Paris est la capitale de la France.\n\n"
    "Exemple :\n"
    "Contexte : Le fer fond à 1538 °C.\n"
    "Question : Quel est le point de fusion de l'or ?\n"
    "Réponse : Information non disponible."
)

JUDGE_FAITHFULNESS = (
    "Tu es un évaluateur de fidélité strict mais juste. "
    "Analyse la réponse et détermine si chaque affirmation est dérivée du contexte. "
    "Une reformulation ou paraphrase du contexte est ACCEPTÉE comme fidèle. "
    "Seules les informations inventées ou absentes du contexte sont NON fidèles. "
    "Réponds UNIQUEMENT en JSON : "
    "{\"total_claims\": N, \"supported_claims\": M, \"faithfulness\": M/N}"
)

JUDGE_RELEVANCE = (
    "Tu évalues la pertinence d'une réponse. La réponse répond-elle "
    "directement et complètement à la question ? "
    "Réponds en JSON : {\"relevance_score\": float entre 0 et 1}"
)

JUDGE_CONTEXT_PRECISION = (
    "Tu évalues la précision du contexte. Le contexte fourni contient-il "
    "l'information nécessaire et suffisante pour répondre à la question ? "
    "Réponds en JSON : {\"context_precision\": float entre 0 et 1}"
)


def _safe_float(raw: str, key: str) -> float:
    """从 JSON 回答中稳健地提取浮点数。"""
    m = re.search(r'"' + key + r'"\s*:\s*([\d.]+)', raw)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            pass
    # 退回到搜索任意浮点数
    floats = re.findall(r"0\.\d+|1\.0|1(?:\.\d+)?", raw)
    if floats:
        return min(1.0, max(0.0, float(floats[0])))
    return 0.0


# ── 程序入口 ────────────────────────────────────────────────────────────

def run(n_samples: int | None = None) -> dict[str, Any]:
    data = load_rag_data(n_samples) if n_samples else load_rag_data()
    logger.info("=== Exp 4 : RAG Faithfulness (RAGAS) — %d échantillons ===", len(data))

    # ── 从检查点继续 ────────────────────────────────────────────────
    ckpt = load_checkpoint("rag")
    if ckpt:
        start_idx = ckpt["next_idx"]
        faithfulness_scores = ckpt["faithfulness_scores"]
        relevance_scores = ckpt["relevance_scores"]
        context_scores = ckpt["context_scores"]
        details = ckpt["details"]
        elapsed_prev = ckpt.get("elapsed", 0.0)
        logger.info("Reprise RAG à idx=%d", start_idx)
    else:
        start_idx = 0
        faithfulness_scores = []
        relevance_scores = []
        context_scores = []
        details = []
        elapsed_prev = 0.0

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        question = sample["question"]
        context = sample["context"]

        # ── 步骤 1：模型根据上下文生成答案 ────────────────────────────
        answer_prompt = (
            f"Contexte :\n{context}\n\n"
            f"Question : {question}\n\n"
            "Réponds en utilisant UNIQUEMENT le contexte ci-dessus."
        )
        generated_answer = generate(
            answer_prompt, system=ANSWER_SYSTEM, max_new_tokens=512,
        )

        # ── 步骤 2：评估忠实度 ──────────────────────────────────────
        faith_prompt = (
            f"Contexte :\n{context}\n\n"
            f"Réponse du système :\n{generated_answer}\n\n"
            "Évalue la fidélité."
        )
        faith_raw = generate_structured(
            faith_prompt, system=JUDGE_FAITHFULNESS, json_mode=True, max_new_tokens=256,
        )
        faith_score = _safe_float(faith_raw, "faithfulness")

        # ── 双评审：使用不同提示词进行第二次评估 ─────────────────────
        faith_prompt2 = (
            f"Contexte :\n{context}\n\n"
            f"Réponse :\n{generated_answer}\n\n"
            "La réponse contient-elle des informations qui ne sont PAS "
            "dans le contexte ? Réponds en JSON : "
            "{\"has_hallucination\": true/false, \"faithfulness\": 0.0-1.0}"
        )
        faith_raw2 = generate_structured(
            faith_prompt2, system=JUDGE_FAITHFULNESS, json_mode=True, max_new_tokens=256,
        )
        faith_score2 = _safe_float(faith_raw2, "faithfulness")

        # ── 结构化检查：拆分主张并进行自然语言推断 ──────────────────
        checker_result = compute_faithfulness(generated_answer, context)
        faith_score3 = checker_result["faithfulness"]

        # 取两位评审和结构化检查器三个分数的平均值
        faith_score = round((faith_score + faith_score2 + faith_score3) / 3, 4)
        faithfulness_scores.append(faith_score)

        # ── 步骤 3：评估答案相关性 ──────────────────────────────────
        rel_prompt = (
            f"Question : {question}\n\n"
            f"Réponse : {generated_answer}\n\n"
            "Évalue la pertinence."
        )
        rel_raw = generate_structured(
            rel_prompt, system=JUDGE_RELEVANCE, json_mode=True, max_new_tokens=256,
        )
        rel_score = _safe_float(rel_raw, "relevance_score")
        relevance_scores.append(rel_score)

        # ── 步骤 4：评估上下文精确率 ────────────────────────────────
        ctx_prompt = (
            f"Question : {question}\n\n"
            f"Contexte fourni :\n{context}\n\n"
            "Évalue la précision du contexte."
        )
        ctx_raw = generate_structured(
            ctx_prompt, system=JUDGE_CONTEXT_PRECISION, json_mode=True, max_new_tokens=256,
        )
        ctx_score = _safe_float(ctx_raw, "context_precision")
        context_scores.append(ctx_score)

        details.append({
            "idx": i,
            "question": question,
            "generated_answer": generated_answer[:300],
            "faithfulness": round(faith_score, 3),
            "checker_faithfulness": round(faith_score3, 3),
            "checker_claims": checker_result["total_claims"],
            "checker_supported": checker_result["supported"],
            "relevance": round(rel_score, 3),
            "context_precision": round(ctx_score, 3),
        })

        if (i + 1) % 5 == 0:
            logger.info("  … %d/%d traités", i + 1, len(data))
            save_checkpoint("rag", {
                "next_idx": i + 1,
                "faithfulness_scores": faithfulness_scores,
                "relevance_scores": relevance_scores,
                "context_scores": context_scores,
                "details": details,
                "elapsed": elapsed_prev + (time.time() - t0),
            })

    elapsed = elapsed_prev + (time.time() - t0)

    n = len(data)
    avg_faith = sum(faithfulness_scores) / n if n else 0.0
    avg_rel = sum(relevance_scores) / n if n else 0.0
    avg_ctx = sum(context_scores) / n if n else 0.0

    results = {
        "experiment": "rag_faithfulness",
        "n_samples": n,
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevance": round(avg_rel, 4),
        "avg_context_precision": round(avg_ctx, 4),
        "faithfulness_ci": bootstrap_ci(faithfulness_scores),
        "target_faithfulness": TARGET_FAITHFULNESS,
        "target_met": avg_faith >= 0.85,
        "elapsed_seconds": round(elapsed, 1),
        "details": details,
    }
    logger.info(
        "RAGAS — Faith=%.3f  Relevance=%.3f  CtxPrec=%.3f  [%.1fs]",
        avg_faith, avg_rel, avg_ctx, elapsed,
    )
    clear_checkpoint("rag")
    return results
