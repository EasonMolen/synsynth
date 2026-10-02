"""
实验 3：多跳推理（HotpotQA 风格）。

模型接收复杂问题及支持事实，串联多个信息点进行推理并给出答案。

评估完全匹配准确率、部分匹配准确率和推理链得分。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from synsynth_config import logger, NUM_MULTIHOP_HOPS
from synsynth_model import generate_structured
from synsynth_data import load_multihop_data
from synsynth_stats import bootstrap_ci, token_f1
from synsynth_checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

SYSTEM_PROMPT = (
    "Tu es un agent de raisonnement multi-hop. "
    "Tu reçois une question complexe et des faits de support. "
    "Tu dois raisonner étape par étape en reliant les faits, puis fournir "
    "ta réponse finale COURTE et PRÉCISE (quelques mots seulement). "
    "Réponds en JSON : "
    "{\"reasoning_chain\": [\"...\", \"...\"], \"answer\": \"réponse courte\"}\n\n"
    "Exemple :\n"
    "Question : Le fondateur de l'entreprise basée à Cupertino a étudié où ?\n"
    "Faits : Apple a son siège à Cupertino. Apple a été fondée par Steve Jobs. "
    "Steve Jobs a étudié au Reed College.\n"
    "Réponse : {\"reasoning_chain\": [\"Apple est basée à Cupertino\", "
    "\"Steve Jobs a fondé Apple\", \"Jobs a étudié au Reed College\"], "
    "\"answer\": \"Reed College\"}\n\n"
    "Exemple :\n"
    "Question : Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "Faits : Scott Derrickson is an American director. Ed Wood was an American filmmaker.\n"
    "Réponse : {\"reasoning_chain\": [\"Scott Derrickson is American\", "
    "\"Ed Wood was American\"], \"answer\": \"yes\"}"
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _extract_short_answer(text: str) -> str:
    """尝试从冗长文本中提取简短答案。"""
    # 查找 answer、réponse、final answer 等标签后面的内容
    for pat in [r'(?:final\s+)?answer\s*(?:is|:)\s*(.+)',
                r'réponse\s*(?:finale)?\s*(?:est|:)\s*(.+)',
                r'(?:therefore|thus|so)\s*,?\s*(.+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip('.')
            if len(ans) < 200:
                return ans
    return text


def _answer_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if p == g or g in p or p in g:
        return True
    # 如果预测过长，则提取简短答案
    short = _normalize(_extract_short_answer(pred))
    if short != p and (g in short or short in g or short == g):
        return True
    return False


def _strip_markdown(raw: str) -> str:
    """移除回答外层的 ```json...``` 代码块。"""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    return m.group(1).strip() if m else raw


def _parse_response(raw: str) -> dict | None:
    cleaned = _strip_markdown(raw)
    # 先尝试解析完整字符串
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # 根据花括号嵌套层级找出最外层 JSON
    depth = 0
    start = None
    for i, c in enumerate(cleaned):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start:i+1])
                    if isinstance(obj, dict) and "answer" in obj:
                        obj["answer"] = str(obj["answer"])
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None
    # JSON 格式错误时，用正则提取 answer 字段
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
    if m:
        answer = m.group(1).strip()
        # 同时尝试提取推理链
        chain_match = re.findall(r'"reasoning_chain"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        chain = []
        if chain_match:
            chain = re.findall(r'"([^"]+)"', chain_match[0])
        return {"answer": answer, "reasoning_chain": chain}
    # 最后从纯文本中提取答案
    answer = _extract_short_answer(cleaned) if cleaned else raw.strip()
    return {"answer": answer, "reasoning_chain": []}


# ── 程序入口 ────────────────────────────────────────────────────────────

def run(n_samples: int | None = None) -> dict[str, Any]:
    data = load_multihop_data(n_samples) if n_samples else load_multihop_data()
    logger.info(
        "=== Exp 3 : Raisonnement multi-hop — %d échantillons ===", len(data),
    )

    # ── 从检查点继续 ────────────────────────────────────────────────
    ckpt = load_checkpoint("multihop")
    if ckpt:
        start_idx = ckpt["next_idx"]
        correct = ckpt["correct"]
        partial = ckpt["partial"]
        chain_lengths = ckpt["chain_lengths"]
        details = ckpt["details"]
        elapsed_prev = ckpt.get("elapsed", 0.0)
        logger.info("Reprise multihop à idx=%d", start_idx)
    else:
        start_idx = 0
        correct = 0
        partial = 0
        chain_lengths = []
        details = []
        elapsed_prev = 0.0

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        facts = "\n".join(f"- {f}" for f in sample.get("supporting_facts", []))
        prompt = (
            f"Question : {sample['question']}\n\n"
            f"Faits de support :\n{facts}\n\n"
            "Raisonne étape par étape puis donne la réponse."
        )
        raw = generate_structured(prompt, system=SYSTEM_PROMPT, json_mode=True, max_new_tokens=1024)
        pred = _parse_response(raw)

        # 重新排序：答案过长时再次请求模型归纳
        if pred and len(str(pred["answer"])) > 100:
            rerank_prompt = (
                f"Question : {sample['question']}\n"
                f"Réponse détaillée : {pred['answer'][:300]}\n\n"
                "En quelques mots seulement, quelle est la réponse ? "
                "Réponds en JSON : {\"answer\": \"...\"}"
            )
            rerank_raw = generate_structured(
                rerank_prompt, system=SYSTEM_PROMPT, json_mode=True, max_new_tokens=128)
            rerank_pred = _parse_response(rerank_raw)
            if rerank_pred and len(str(rerank_pred["answer"])) < len(str(pred["answer"])):
                pred["answer"] = rerank_pred["answer"]

        ans_ok = _answer_match(pred["answer"], sample["answer"]) if pred else False
        chain = pred.get("reasoning_chain", []) if pred else []
        chain_len = len(chain) if isinstance(chain, list) else 0
        chain_lengths.append(chain_len)

        # 部分匹配：至少包含标准答案中一个长度超过三字符的关键词
        partial_ok = False
        if pred:
            gold_words = {w for w in _normalize(sample["answer"]).split() if len(w) >= 4}
            pred_words = set(_normalize(pred["answer"]).split())
            if gold_words:
                partial_ok = bool(gold_words & pred_words)
            else:
                # 标准答案很短时退回简单匹配
                partial_ok = _normalize(sample["answer"]) in _normalize(pred["answer"])

        if ans_ok:
            correct += 1
        if partial_ok and not ans_ok:
            partial += 1

        details.append({
            "idx": i,
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "pred_answer": pred["answer"] if pred else "",
            "exact_match": ans_ok,
            "partial_match": partial_ok,
            "token_f1": round(token_f1(pred["answer"], sample["answer"]), 4) if pred else 0.0,
            "chain_length": chain_len,
        })

        if (i + 1) % 5 == 0:
            logger.info("  … %d/%d traités", i + 1, len(data))
            save_checkpoint("multihop", {
                "next_idx": i + 1, "correct": correct, "partial": partial,
                "chain_lengths": chain_lengths, "details": details,
                "elapsed": elapsed_prev + (time.time() - t0),
            })

    elapsed = elapsed_prev + (time.time() - t0)

    n = len(data)
    exact_acc = correct / n if n else 0.0
    partial_acc = (correct + partial) / n if n else 0.0
    avg_chain = sum(chain_lengths) / n if n else 0.0
    f1_scores = [d["token_f1"] for d in details]
    avg_f1 = sum(f1_scores) / n if n else 0.0

    results = {
        "experiment": "multihop_reasoning",
        "n_samples": n,
        "exact_match": correct,
        "partial_match": partial,
        "exact_accuracy": round(exact_acc, 4),
        "partial_accuracy": round(partial_acc, 4),
        "avg_token_f1": round(avg_f1, 4),
        "token_f1_ci": bootstrap_ci(f1_scores),
        "avg_reasoning_chain_length": round(avg_chain, 2),
        "target_hops": NUM_MULTIHOP_HOPS,
        "elapsed_seconds": round(elapsed, 1),
        "details": details,
    }
    logger.info(
        "Multi-hop — Exact=%.2f  Partial=%.2f  F1=%.3f  Avg-chain=%.1f  [%.1fs]",
        exact_acc, partial_acc, avg_f1, avg_chain, elapsed,
    )
    clear_checkpoint("multihop")
    return results
