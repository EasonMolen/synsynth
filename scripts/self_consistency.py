#!/usr/bin/env python3
"""
D2：模型内部自一致性实验（Wang 等，2023）。

每个模型在温度 T>0 时生成 k=5 个答案，再进行多数投票。
结果与跨模型投票（H6）及 T≈0 的单次运行比较。

用法：
    python scripts/self_consistency.py                        # 排名前三的模型
    python scripts/self_consistency.py --models phi4:latest    # 单个模型
    python scripts/self_consistency.py --k 3 --temp 0.5       # 自定义参数
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter

# ── 路径 ────────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFICULT_PATH = os.path.join(WORKSPACE, "results", "model_selection",
                              "difficult_questions.json")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "self_consistency")
BENCHMARK_DIR = os.path.join(WORKSPACE, "results", "model_selection")
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
from synsynth_stats import token_f1

# ── 配置 ────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://127.0.0.1:11434"
TIMEOUT = 600
DEFAULT_K = 5
DEFAULT_TEMP = 0.7

# 基准测试中 181 道难题完全匹配率最高的三个模型
DEFAULT_MODELS = [
    "gpt-oss:20b",           # EM=0.1768，表现最佳
    "phi4-reasoning:plus",   # EM=0.1657
    "phi4:latest",           # EM=0.1326
]

# ── 系统提示词，与基准测试一致 ────────────────────────────────────────
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


# ── 解析回答，与基准测试一致 ──────────────────────────────────────────

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
        answer = m.group(1).strip()
        return {"answer": answer, "reasoning_chain": []}

    answer = _extract_short_answer(cleaned)
    return {"answer": answer, "reasoning_chain": []}


# ── 答案匹配，与基准测试一致 ──────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def answer_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if p == g or g in p or p in g:
        return True
    short = _normalize(_extract_short_answer(pred))
    if short != p and (g in short or short in g or short == g):
        return True
    return False


# ── 多数投票 ────────────────────────────────────────────────────────────

def majority_vote(answers: list[str]) -> str:
    """对归一化后的答案进行多数投票。"""
    if not answers:
        return ""
    normalized = [_normalize(a) for a in answers]
    counts = Counter(normalized)
    winner = counts.most_common(1)[0][0]
    # 返回票数最高答案的原始写法
    for a, n in zip(answers, normalized):
        if n == winner:
            return a
    return answers[0]


def vote_agreement(answers: list[str]) -> float:
    """计算与投票结果相同的答案比例。"""
    if not answers:
        return 0.0
    winner = _normalize(majority_vote(answers))
    normalized = [_normalize(a) for a in answers]
    return sum(1 for n in normalized if n == winner) / len(normalized)


# ── 数据 ────────────────────────────────────────────────────────────────

def load_difficult_questions() -> list[dict]:
    with open(DIFFICULT_PATH) as f:
        return json.load(f)


def load_supporting_facts() -> dict[int, list[str]]:
    from synsynth_data import load_multihop_data
    data = load_multihop_data(500)
    return {i: d.get("supporting_facts", []) for i, d in enumerate(data)}


def load_baseline_results(model: str) -> dict | None:
    """加载基准测试的单次运行结果用于比较。"""
    safe = model.replace(":", "_").replace("/", "_")
    path = os.path.join(BENCHMARK_DIR, f"benchmark_{safe}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── 自一致性评估 ────────────────────────────────────────────────────────

def evaluate_self_consistency(model: str, questions: list[dict],
                              facts_by_idx: dict[int, list[str]],
                              k: int, temperature: float) -> dict:
    safe = model.replace(":", "_").replace("/", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f"sc_{safe}_k{k}.json")

    details = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        start_idx = len(details)
        if start_idx >= len(questions):
            print(f"  [SKIP] {model} — déjà terminé ({start_idx}/{len(questions)})")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Self-consistency: {model} (k={k}, T={temperature})")
    print(f"{'='*60}")

    t0 = time.time()
    for qi in range(start_idx, len(questions)):
        q = questions[qi]
        idx = q["idx"]
        facts = facts_by_idx.get(idx, [])
        facts_str = "\n".join(f"- {f}" for f in facts)
        prompt = (
            f"Question : {q['question']}\n\n"
            f"Faits de support :\n{facts_str}\n\n"
            "Raisonne étape par étape puis donne la réponse."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # 在 T>0 时独立调用 k 次
        raw_answers = []
        parsed_answers = []
        for ki in range(k):
            raw = ollama_chat(model, messages,
                              temperature=temperature,
                              json_format=True)
            pred = parse_response(raw)
            ans = pred["answer"] if pred else ""
            raw_answers.append(raw[:500])  # 截断以便存储
            parsed_answers.append(ans)

        # 多数投票
        voted = majority_vote(parsed_answers)
        agreement = vote_agreement(parsed_answers)

        gold = q["gold_answer"]
        em_voted = answer_match(voted, gold)
        f1_voted = token_f1(voted, gold)

        # 为便于比较，计算每个单独答案的完全匹配率
        individual_ems = [answer_match(a, gold) for a in parsed_answers]
        any_correct = any(individual_ems)
        all_correct = all(individual_ems)

        details.append({
            "idx": idx,
            "question": q["question"],
            "gold_answer": gold,
            "k_answers": parsed_answers,
            "voted_answer": voted,
            "agreement": round(agreement, 2),
            "em_voted": em_voted,
            "f1_voted": round(f1_voted, 4),
            "individual_ems": individual_ems,
            "any_correct": any_correct,
        })

        if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
            elapsed = time.time() - t0
            n_done = qi + 1
            em_agg = sum(1 for d in details if d["em_voted"]) / len(details)
            oracle = sum(1 for d in details if d["any_correct"]) / len(details)
            agree_avg = sum(d["agreement"] for d in details) / len(details)
            print(f"  {model}: {n_done}/{len(questions)} "
                  f"(EM_vote={em_agg:.3f}, Oracle_k={oracle:.3f}, "
                  f"Agree={agree_avg:.2f}, {elapsed:.0f}s)")

            result = _compile_results(details, model, k, temperature)
            with open(ckpt_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compile_results(details, model, k, temperature)


def _compile_results(details: list[dict], model: str,
                     k: int, temperature: float) -> dict:
    n = len(details)
    if n == 0:
        return {"model": model, "k": k, "temperature": temperature,
                "n": 0, "details": []}

    em_voted = sum(1 for d in details if d["em_voted"]) / n
    f1_voted = sum(d["f1_voted"] for d in details) / n
    oracle_k = sum(1 for d in details if d["any_correct"]) / n
    agree_avg = sum(d["agreement"] for d in details) / n

    # 完全一致表示所有答案相同
    perfect_agree = sum(1 for d in details if d["agreement"] == 1.0) / n

    return {
        "model": model,
        "k": k,
        "temperature": temperature,
        "n": n,
        "em_voted": round(em_voted, 4),
        "f1_voted": round(f1_voted, 4),
        "oracle_k": round(oracle_k, 4),
        "agreement_avg": round(agree_avg, 4),
        "perfect_agreement_pct": round(perfect_agree * 100, 1),
        "details": details,
    }


# ── 比较汇总 ──────────────────────────────────────────────────────────

def print_comparison(results: list[dict]):
    print("\n" + "="*80)
    print("SYNTHÈSE D2 — Self-consistency vs baselines")
    print("="*80)

    header = (f"{'Modèle':<25} {'EM_T≈0':>7} {'EM_SC':>7} "
              f"{'Δ':>6} {'Oracle_k':>8} {'Agree':>6}")
    print(header)
    print("-" * 80)

    for r in results:
        model = r["model"]
        baseline = load_baseline_results(model)
        em_base = baseline["em"] if baseline else "—"
        em_sc = r["em_voted"]
        delta = f"+{em_sc - em_base:.4f}" if isinstance(em_base, float) else "—"
        em_base_str = f"{em_base:.4f}" if isinstance(em_base, float) else em_base
        print(f"{model:<25} {em_base_str:>7} {em_sc:.4f} "
              f"{delta:>6} {r['oracle_k']:.4f} {r['agreement_avg']:.2f}")

    print()
    # 加载跨模型投票结果（H6）进行比较
    h6_path = os.path.join(BENCHMARK_DIR, "benchmark_summary_difficult.json")
    if os.path.exists(h6_path):
        with open(h6_path) as f:
            h6 = json.load(f)
        oracle_inter = h6.get("oracle_em", "?")
        print(f"Vote inter-modèles (H6) : Oracle = {oracle_inter}")
        print(f"Meilleur SC EM :           {max(r['em_voted'] for r in results):.4f}")
        print(f"Meilleur SC Oracle_k :     {max(r['oracle_k'] for r in results):.4f}")


def save_summary(results: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []
    for r in results:
        baseline = load_baseline_results(r["model"])
        entry = {
            "model": r["model"],
            "k": r["k"],
            "temperature": r["temperature"],
            "n_questions": r["n"],
            "em_self_consistency": r["em_voted"],
            "f1_self_consistency": r["f1_voted"],
            "oracle_k": r["oracle_k"],
            "agreement_avg": r["agreement_avg"],
            "perfect_agreement_pct": r["perfect_agreement_pct"],
            "em_baseline_T0": baseline["em"] if baseline else None,
            "f1_baseline_T0": baseline["f1"] if baseline else None,
        }
        summary.append(entry)

    path = os.path.join(RESULTS_DIR, "self_consistency_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary → {path}")


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D2: Self-consistency intra-modèle")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Modèles à évaluer")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Nombre de réponses par question (default: 5)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Température (default: 0.7)")
    args = parser.parse_args()

    print(f"D2 Self-consistency — k={args.k}, T={args.temp}")
    print(f"Modèles : {args.models}")

    questions = load_difficult_questions()
    print(f"Questions difficiles : {len(questions)}")

    print("Chargement des faits de support…")
    facts_by_idx = load_supporting_facts()
    print(f"Faits chargés pour {len(facts_by_idx)} questions.")

    results = []
    for model in args.models:
        result = evaluate_self_consistency(
            model, questions, facts_by_idx,
            k=args.k, temperature=args.temp,
        )
        results.append(result)

    print_comparison(results)
    save_summary(results)


if __name__ == "__main__":
    main()
