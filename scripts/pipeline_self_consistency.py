#!/usr/bin/env python3
"""
V5a：将自一致性方法集成到多跳推理流程。

在温度 T>0 时生成 k 个答案并进行多数投票，
再与 V4 的单次调用流程（T≈0.1）比较。

模式：
  - 基础自一致性：k=3 或 k=5 的多数投票。
  - 自一致性加基于一致程度的路由（V5b 预览版）。

用法：
    # 对流程中的 500 道题进行 k=3 的自一致性评估
    python scripts/pipeline_self_consistency.py

    # k=5，单个模型
    python scripts/pipeline_self_consistency.py --k 5 --model phi4:latest

    # 比较三个模型
    python scripts/pipeline_self_consistency.py --models phi4:latest gpt-oss:20b phi4-reasoning:plus

    # 级联模式（V5b）：按答案一致程度路由
    python scripts/pipeline_self_consistency.py --cascade --k 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from synsynth_config import logger, RESULTS_DIR
from synsynth_data import load_multihop_data
from synsynth_stats import bootstrap_ci, token_f1
from synsynth_checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

# ── 直接调用 Ollama，以便控制温度 T ───────────────────────────────────
import urllib.request
import urllib.error

OLLAMA_BASE = "http://127.0.0.1:11434"
TIMEOUT = 600

SYSTEM_PROMPT = (
    "Tu es un agent de raisonnement multi-hop. "
    "Tu reçois une question complexe et des faits de support. "
    "Tu dois raisonner étape par étape en reliant les faits, puis fournir "
    "ta réponse finale COURTE et PRÉCISE (quelques mots seulement). "
    "Réponds en JSON : "
    '{"reasoning_chain": ["...", "..."], "answer": "réponse courte"}\n\n'
    "Exemple :\n"
    "Question : Le fondateur de l'entreprise basée à Cupertino a étudié où ?\n"
    "Faits : Apple a son siège à Cupertino. Apple a été fondée par Steve Jobs. "
    "Steve Jobs a étudié au Reed College.\n"
    'Réponse : {"reasoning_chain": ["Apple est basée à Cupertino", '
    '"Steve Jobs a fondé Apple", "Jobs a étudié au Reed College"], '
    '"answer": "Reed College"}\n\n'
    "Exemple :\n"
    "Question : Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "Faits : Scott Derrickson is an American director. Ed Wood was an American filmmaker.\n"
    'Réponse : {"reasoning_chain": ["Scott Derrickson is American", '
    '"Ed Wood was American"], "answer": "yes"}'
    "\n\nTu dois répondre UNIQUEMENT avec un objet JSON valide, "
    "sans texte avant ni après."
)

# ── 流程默认参数 ────────────────────────────────────────────────────────
DEFAULT_MODEL = "phi4:latest"       # V4 流程中表现最好的多跳推理模型
CASCADE_MODELS = [
    "phi4:latest",              # 主模型，流程中的完全匹配率最高
    "gpt-oss:20b",             # 备用模型，自一致性表现最佳的互补 MoE 模型
]
DEFAULT_K = 3
DEFAULT_TEMP = 0.7

RESULTS_SC_DIR = os.path.join(RESULTS_DIR, "pipeline_self_consistency")


# ── 调用 Ollama ─────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list[dict],
                temperature: float = 0.7,
                json_format: bool = False) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 4096,
            "num_ctx": 8192,
        },
    }
    if json_format:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("message", {}).get("content", "")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [ERREUR Ollama] {e}")
        return ""


# ── 回答解析与匹配，与主流程一致 ──────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _strip_markdown(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    return m.group(1).strip() if m else raw


def _extract_short_answer(text: str) -> str:
    for pat in [r'(?:final\s+)?answer\s*(?:is|:)\s*(.+)',
                r'réponse\s*(?:finale)?\s*(?:est|:)\s*(.+)',
                r'(?:therefore|thus|so)\s*,?\s*(.+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip('.')
            if len(ans) < 200:
                return ans
    return text


def parse_response(raw: str) -> dict | None:
    if not raw.strip():
        return None
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw
    cleaned = _strip_markdown(cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            obj["answer"] = str(obj["answer"])
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
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
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
    if m:
        return {"answer": m.group(1).strip(), "reasoning_chain": []}
    answer = _extract_short_answer(cleaned)
    return {"answer": answer, "reasoning_chain": []}


def answer_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if p == g or g in p or p in g:
        return True
    short = _normalize(_extract_short_answer(pred))
    if short != p and (g in short or short in g or short == g):
        return True
    return False


# ── 投票与一致程度 ─────────────────────────────────────────────────────

def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized = [_normalize(a) for a in answers]
    counts = Counter(normalized)
    winner = counts.most_common(1)[0][0]
    for a, n in zip(answers, normalized):
        if n == winner:
            return a
    return answers[0]


def vote_agreement(answers: list[str]) -> float:
    if not answers:
        return 0.0
    winner = _normalize(majority_vote(answers))
    normalized = [_normalize(a) for a in answers]
    return sum(1 for n in normalized if n == winner) / len(normalized)


# ── V4 基线：温度约为 0.1 的单次调用 ─────────────────────────────────

def load_v4_baseline() -> dict | None:
    """加载已有的多跳推理 V4 结果用于比较。

    优先选择 V4 学习曲线中表现最好的训练样本数。
    """
    # 取 V4 学习曲线中的最高完全匹配率
    lc_path = os.path.join(RESULTS_DIR, "learning_curve",
                           "learning_curve_multihop_v4.json")
    if os.path.exists(lc_path):
        with open(lc_path) as f:
            lc = json.load(f)
        if isinstance(lc, list) and lc:
            best = max(lc, key=lambda x: x.get("exact_accuracy", 0))
            return best

    # 找不到时使用 all_results.json
    all_path = os.path.join(RESULTS_DIR, "all_results.json")
    if os.path.exists(all_path):
        with open(all_path) as f:
            data = json.load(f)
        if "multihop_reasoning_qlora" in data:
            return data["multihop_reasoning_qlora"]

    return None


# ── 对完整流程进行自一致性评估 ────────────────────────────────────────

def evaluate_pipeline_sc(model: str, data: list[dict],
                         k: int, temperature: float) -> dict:
    """在流程中的全部问题上评估自一致性。"""
    safe = model.replace(":", "_").replace("/", "_")
    os.makedirs(RESULTS_SC_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_SC_DIR, f"sc_{safe}_k{k}.json")

    details = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        start_idx = len(details)
        if start_idx >= len(data):
            print(f"  [SKIP] {model} — déjà terminé ({start_idx}/{len(data)})")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Pipeline SC: {model} (k={k}, T={temperature})")
    print(f"  Reprise à {start_idx}/{len(data)}")
    print(f"{'='*60}")

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        facts = "\n".join(f"- {f}" for f in sample.get("supporting_facts", []))
        prompt = (
            f"Question : {sample['question']}\n\n"
            f"Faits de support :\n{facts}\n\n"
            "Raisonne étape par étape puis donne la réponse."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # 在 T>0 时独立调用 k 次
        parsed_answers = []
        for ki in range(k):
            raw = ollama_chat(model, messages,
                              temperature=temperature,
                              json_format=True)
            pred = parse_response(raw)
            ans = pred["answer"] if pred else ""
            parsed_answers.append(ans)

        # 多数投票
        voted = majority_vote(parsed_answers)
        agreement = vote_agreement(parsed_answers)
        gold = sample["answer"]

        em_voted = answer_match(voted, gold)
        f1_voted = token_f1(voted, gold)
        individual_ems = [answer_match(a, gold) for a in parsed_answers]

        # 单次调用使用 T≈0.1，而这里的第一次调用使用 T=0.7；
        # 因此与 V4 基线比较。

        details.append({
            "idx": i,
            "question": sample["question"],
            "gold_answer": gold,
            "k_answers": parsed_answers,
            "voted_answer": voted,
            "agreement": round(agreement, 2),
            "em_voted": em_voted,
            "f1_voted": round(f1_voted, 4),
            "individual_ems": individual_ems,
            "any_correct": any(individual_ems),
        })

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            elapsed = time.time() - t0
            n_done = i + 1
            em_agg = sum(1 for d in details if d["em_voted"]) / len(details)
            oracle = sum(1 for d in details
                         if d["any_correct"]) / len(details)
            agree_avg = sum(d["agreement"] for d in details) / len(details)
            print(f"  {model}: {n_done}/{len(data)} "
                  f"(EM_vote={em_agg:.3f}, Oracle={oracle:.3f}, "
                  f"Agree={agree_avg:.2f}, {elapsed:.0f}s)")

            result = _compile(details, model, k, temperature)
            with open(ckpt_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compile(details, model, k, temperature)


# ── 级联 V5b：根据答案一致程度路由 ──────────────────────────────────

def evaluate_pipeline_cascade(models: list[str], data: list[dict],
                              k: int, temperature: float,
                              threshold_high: float = 0.8,
                              threshold_low: float = 0.4) -> dict:
    """级联流程：先用主模型，一致程度较低时再用次模型。

    策略参考 D2 的一致性悖论：
    - 一致度 ≥ threshold_high：接受投票结果。
    - 一致度在 [threshold_low, threshold_high) 内：转到第二个模型。
    - 一致度 < threshold_low：标记为不确定，仍尝试第二个模型。
    """
    if len(models) < 2:
        raise ValueError("La cascade nécessite au moins 2 modèles")

    primary, fallback = models[0], models[1]
    os.makedirs(RESULTS_SC_DIR, exist_ok=True)
    safe_p = primary.replace(":", "_").replace("/", "_")
    safe_f = fallback.replace(":", "_").replace("/", "_")
    ckpt_path = os.path.join(
        RESULTS_SC_DIR, f"cascade_{safe_p}_{safe_f}_k{k}.json")

    details = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        start_idx = len(details)
        if start_idx >= len(data):
            print(f"  [SKIP] cascade — déjà terminé")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Pipeline CASCADE: {primary} → {fallback}")
    print(f"  k={k}, T={temperature}, "
          f"seuils=[{threshold_low}, {threshold_high}]")
    print(f"  Reprise à {start_idx}/{len(data)}")
    print(f"{'='*60}")

    stats = {"primary_accepted": 0, "rerouted": 0, "uncertain": 0}
    t0 = time.time()

    for i in range(start_idx, len(data)):
        sample = data[i]
        facts = "\n".join(f"- {f}" for f in sample.get("supporting_facts", []))
        prompt = (
            f"Question : {sample['question']}\n\n"
            f"Faits de support :\n{facts}\n\n"
            "Raisonne étape par étape puis donne la réponse."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        gold = sample["answer"]

        # 阶段 1：主模型生成 k 个答案
        primary_answers = []
        for ki in range(k):
            raw = ollama_chat(primary, messages,
                              temperature=temperature, json_format=True)
            pred = parse_response(raw)
            primary_answers.append(pred["answer"] if pred else "")

        primary_voted = majority_vote(primary_answers)
        primary_agreement = vote_agreement(primary_answers)

        # 决定是否路由到次模型
        used_model = primary
        final_voted = primary_voted
        final_agreement = primary_agreement
        final_answers = primary_answers
        fallback_answers = []
        route = "primary"

        if primary_agreement < threshold_high:
            # 路由到备用模型
            route = "rerouted" if primary_agreement >= threshold_low else "uncertain"
            fallback_answers = []
            for ki in range(k):
                raw = ollama_chat(fallback, messages,
                                  temperature=temperature, json_format=True)
                pred = parse_response(raw)
                fallback_answers.append(pred["answer"] if pred else "")

            fallback_voted = majority_vote(fallback_answers)
            fallback_agreement = vote_agreement(fallback_answers)

            # 选择一致程度最高的答案
            if fallback_agreement > primary_agreement:
                final_voted = fallback_voted
                final_agreement = fallback_agreement
                final_answers = fallback_answers
                used_model = fallback
            # 否则保留主模型的答案，即使一致程度较低

        stats[route if route != "primary" else "primary_accepted"] += 1

        em_voted = answer_match(final_voted, gold)
        f1_voted = token_f1(final_voted, gold)
        all_answers = primary_answers + fallback_answers
        any_correct = any(answer_match(a, gold) for a in all_answers if a)

        details.append({
            "idx": i,
            "question": sample["question"],
            "gold_answer": gold,
            "primary_answers": primary_answers,
            "primary_agreement": round(primary_agreement, 2),
            "fallback_answers": fallback_answers,
            "route": route,
            "used_model": used_model,
            "voted_answer": final_voted,
            "agreement": round(final_agreement, 2),
            "em_voted": em_voted,
            "f1_voted": round(f1_voted, 4),
            "any_correct": any_correct,
        })

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            elapsed = time.time() - t0
            n_done = i + 1
            em_agg = sum(1 for d in details if d["em_voted"]) / len(details)
            oracle = sum(1 for d in details
                         if d["any_correct"]) / len(details)
            n_rerouted = sum(1 for d in details if d["route"] != "primary")
            print(f"  cascade: {n_done}/{len(data)} "
                  f"(EM={em_agg:.3f}, Ora={oracle:.3f}, "
                  f"rerouted={n_rerouted}/{n_done}, {elapsed:.0f}s)")

            result = _compile_cascade(
                details, models, k, temperature,
                threshold_high, threshold_low, stats)
            with open(ckpt_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compile_cascade(details, models, k, temperature,
                            threshold_high, threshold_low, stats)


# ── 汇总结果 ────────────────────────────────────────────────────────────

def _compile(details: list[dict], model: str,
             k: int, temperature: float) -> dict:
    n = len(details)
    if n == 0:
        return {"model": model, "k": k, "temperature": temperature,
                "n": 0, "details": []}
    em = sum(1 for d in details if d["em_voted"]) / n
    f1 = sum(d["f1_voted"] for d in details) / n
    oracle = sum(1 for d in details if d["any_correct"]) / n
    agree = sum(d["agreement"] for d in details) / n
    em_ci = bootstrap_ci([1 if d["em_voted"] else 0 for d in details])
    f1_ci = bootstrap_ci([d["f1_voted"] for d in details])
    oracle_ci = bootstrap_ci([1 if d["any_correct"] else 0 for d in details])
    return {
        "model": model, "k": k, "temperature": temperature,
        "n": n,
        "em_voted": round(em, 4),
        "em_ci": [em_ci["ci_low"], em_ci["ci_high"]],
        "f1_voted": round(f1, 4),
        "f1_ci": [f1_ci["ci_low"], f1_ci["ci_high"]],
        "oracle_k": round(oracle, 4),
        "oracle_ci": [oracle_ci["ci_low"], oracle_ci["ci_high"]],
        "agreement_avg": round(agree, 4),
        "details": details,
    }


def _compile_cascade(details: list[dict], models: list[str],
                     k: int, temperature: float,
                     threshold_high: float, threshold_low: float,
                     stats: dict) -> dict:
    n = len(details)
    if n == 0:
        return {"models": models, "k": k, "n": 0, "details": []}
    em = sum(1 for d in details if d["em_voted"]) / n
    f1 = sum(d["f1_voted"] for d in details) / n
    oracle = sum(1 for d in details if d["any_correct"]) / n
    agree = sum(d["agreement"] for d in details) / n
    n_rerouted = sum(1 for d in details if d["route"] != "primary")
    em_ci = bootstrap_ci([1 if d["em_voted"] else 0 for d in details])
    f1_ci = bootstrap_ci([d["f1_voted"] for d in details])
    oracle_ci = bootstrap_ci([1 if d["any_correct"] else 0 for d in details])
    return {
        "mode": "cascade",
        "models": models,
        "k": k, "temperature": temperature,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "n": n,
        "em_voted": round(em, 4),
        "em_ci": [em_ci["ci_low"], em_ci["ci_high"]],
        "f1_voted": round(f1, 4),
        "f1_ci": [f1_ci["ci_low"], f1_ci["ci_high"]],
        "oracle_k": round(oracle, 4),
        "oracle_ci": [oracle_ci["ci_low"], oracle_ci["ci_high"]],
        "agreement_avg": round(agree, 4),
        "routing_stats": {
            "primary_accepted": stats.get("primary_accepted", 0),
            "rerouted": stats.get("rerouted", 0),
            "uncertain": stats.get("uncertain", 0),
            "pct_rerouted": round(n_rerouted / n * 100, 1) if n else 0,
        },
        "details": details,
    }


# ── 展示比较结果 ───────────────────────────────────────────────────────

def print_comparison(results: list[dict]):
    """将自一致性流程与 V4 基线比较。"""
    v4 = load_v4_baseline()
    v4_em = v4.get("exact_accuracy", 0) if v4 else None
    v4_f1 = v4.get("avg_token_f1", 0) if v4 else None
    v4_n = v4.get("n_samples", "?") if v4 else "?"

    print(f"\n{'='*80}")
    print("COMPARAISON — Pipeline V4 (single-shot) vs V5a (self-consistency)")
    print(f"{'='*80}")

    if v4:
        print(f"\n  V4 baseline ({v4.get('experiment','multihop')}, "
              f"N={v4_n}): EM={v4_em:.4f}, F1={v4_f1:.4f}")
    print()

    header = (f"  {'Config':<35} {'EM':>7} {'IC 95%':>15} {'Δ EM':>7} "
              f"{'Oracle':>7} {'IC 95%':>15} {'F1':>7} {'IC 95%':>15} {'Agree':>6}")
    print(header)
    print(f"  {'-'*120}")

    if v4:
        print(f"  {'V4 single-shot T≈0.1':<35} "
              f"{v4_em:>7.4f} {'—':>15} {'—':>7} "
              f"{'—':>7} {'—':>15} "
              f"{v4_f1:>7.4f} {'—':>15} {'—':>6}")

    for r in results:
        label = r.get("mode", "sc")
        if label == "cascade":
            name = f"V5b cascade {r['models'][0]}→{r['models'][1]}"
        else:
            name = f"V5a SC k={r['k']} {r['model']}"
        em = r["em_voted"]
        delta = f"+{em - v4_em:.4f}" if v4_em is not None else "—"
        em_ci = r.get("em_ci", [0, 0])
        f1_ci = r.get("f1_ci", [0, 0])
        oracle_ci = r.get("oracle_ci", [0, 0])
        print(f"  {name:<35} {em:>7.4f} [{em_ci[0]:.3f};{em_ci[1]:.3f}] {delta:>7} "
              f"{r['oracle_k']:>7.4f} [{oracle_ci[0]:.3f};{oracle_ci[1]:.3f}] "
              f"{r['f1_voted']:>7.4f} [{f1_ci[0]:.3f};{f1_ci[1]:.3f}] "
              f"{r['agreement_avg']:>6.2f}")

    print()


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V5a/V5b — Self-consistency intégrée au pipeline")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Modèle principal (default: phi4:latest)")
    parser.add_argument("--models", nargs="+",
                        help="Plusieurs modèles (mode SC simple pour chacun)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Nombre d'échantillons par question (default: 3)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Température (default: 0.7)")
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre de questions (default: toutes = 500)")
    parser.add_argument("--cascade", action="store_true",
                        help="Mode V5b : cascade avec routage par accord")
    parser.add_argument("--cascade-models", nargs=2,
                        default=CASCADE_MODELS,
                        help="Modèles pour la cascade (primary fallback)")
    parser.add_argument("--threshold-high", type=float, default=0.8,
                        help="Seuil d'accord haut pour accepter (default: 0.8)")
    parser.add_argument("--threshold-low", type=float, default=0.4,
                        help="Seuil d'accord bas (default: 0.4)")
    args = parser.parse_args()

    # 加载数据
    data = load_multihop_data(args.n)
    print(f"Questions multi-hop : {len(data)}")

    all_results = []

    if args.cascade:
        # V5b 模式：级联
        result = evaluate_pipeline_cascade(
            args.cascade_models, data,
            k=args.k, temperature=args.temp,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
        )
        all_results.append(result)
    else:
        # V5a 模式：基础自一致性
        models = args.models or [args.model]
        for model in models:
            result = evaluate_pipeline_sc(
                model, data,
                k=args.k, temperature=args.temp,
            )
            all_results.append(result)

    print_comparison(all_results)

    # 保存汇总结果
    os.makedirs(RESULTS_SC_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_SC_DIR, "pipeline_sc_summary.json")
    summary = []
    v4 = load_v4_baseline()
    for r in all_results:
        entry = {
            "mode": r.get("mode", "sc"),
            "model": r.get("model", r.get("models", ["?"])),
            "k": r["k"],
            "n": r["n"],
            "em_voted": r["em_voted"],
            "f1_voted": r["f1_voted"],
            "oracle_k": r["oracle_k"],
            "agreement_avg": r["agreement_avg"],
            "v4_em_baseline": v4.get("exact_accuracy") if v4 else None,
        }
        if "routing_stats" in r:
            entry["routing_stats"] = r["routing_stats"]
        summary.append(entry)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé → {summary_path}")


if __name__ == "__main__":
    main()
