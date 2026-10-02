#!/usr/bin/env python3
"""
自一致性实验（D2）的多次运行方差。

对每个模型独立运行 R 次 D2 实验（在 T>0 时生成 k 个答案并投票），
再计算汇总指标的均值和标准差。

复用 self_consistency.py 中的函数。

用法：
    # 对三个模型各运行五次；中断后可继续
    python scripts/variance_self_consistency.py --runs 5

    # 对单个模型运行三次
    python scripts/variance_self_consistency.py --runs 3 --models gpt-oss:20b

    # 如果所有结果已存在，只做分析而不重新运行
    python scripts/variance_self_consistency.py --runs 5 --analyze-only

    # 第零次运行复用已有的 D2 数据，无需重新计算
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from self_consistency import (
    DEFAULT_K,
    DEFAULT_MODELS,
    DEFAULT_TEMP,
    RESULTS_DIR,
    SYSTEM_PROMPT,
    answer_match,
    load_baseline_results,
    load_difficult_questions,
    load_supporting_facts,
    majority_vote,
    ollama_chat,
    parse_response,
    vote_agreement,
)
from synsynth_stats import token_f1

VARIANCE_DIR = os.path.join(WORKSPACE, "results", "self_consistency", "variance")


# ── 各次运行的文件路径 ──────────────────────────────────────────────────

def _run_path(model: str, k: int, run: int) -> str:
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(VARIANCE_DIR, f"sc_{safe}_k{k}_run{run}.json")


def _original_path(model: str, k: int) -> str:
    """返回第零次运行所用的原始 D2 结果路径。"""
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(RESULTS_DIR, f"sc_{safe}_k{k}.json")


# ── 评估单次运行 ────────────────────────────────────────────────────────

def evaluate_run(model: str, questions: list[dict],
                 facts_by_idx: dict[int, list[str]],
                 k: int, temperature: float, run: int) -> dict:
    """评估完整的一次自一致性运行，并保存检查点。"""
    os.makedirs(VARIANCE_DIR, exist_ok=True)
    ckpt_path = _run_path(model, k, run)

    details = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        start_idx = len(details)
        if start_idx >= len(questions):
            print(f"  [SKIP] {model} run {run} — déjà terminé "
                  f"({start_idx}/{len(questions)})")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Variance run {run}: {model} (k={k}, T={temperature})")
    print(f"  Reprise à {start_idx}/{len(questions)}")
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
        parsed_answers = []
        for ki in range(k):
            raw = ollama_chat(model, messages,
                              temperature=temperature,
                              json_format=True)
            pred = parse_response(raw)
            ans = pred["answer"] if pred else ""
            parsed_answers.append(ans)

        voted = majority_vote(parsed_answers)
        agreement = vote_agreement(parsed_answers)
        gold = q["gold_answer"]
        em_voted = answer_match(voted, gold)
        f1_voted = token_f1(voted, gold)
        individual_ems = [answer_match(a, gold) for a in parsed_answers]

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
            "any_correct": any(individual_ems),
        })

        if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
            elapsed = time.time() - t0
            n_done = qi + 1
            em_agg = sum(1 for d in details if d["em_voted"]) / len(details)
            oracle = sum(1 for d in details
                         if d["any_correct"]) / len(details)
            agree_avg = sum(d["agreement"] for d in details) / len(details)
            print(f"  run {run} | {model}: {n_done}/{len(questions)} "
                  f"(EM={em_agg:.3f}, Ora={oracle:.3f}, "
                  f"Agr={agree_avg:.2f}, {elapsed:.0f}s)")

            result = _compile_run(details, model, k, temperature, run)
            with open(ckpt_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compile_run(details, model, k, temperature, run)


def _compile_run(details: list[dict], model: str,
                 k: int, temperature: float, run: int) -> dict:
    n = len(details)
    if n == 0:
        return {"model": model, "k": k, "temperature": temperature,
                "run": run, "n": 0, "details": []}
    em_voted = sum(1 for d in details if d["em_voted"]) / n
    f1_voted = sum(d["f1_voted"] for d in details) / n
    oracle_k = sum(1 for d in details if d["any_correct"]) / n
    agree_avg = sum(d["agreement"] for d in details) / n
    perfect = sum(1 for d in details if d["agreement"] == 1.0) / n

    return {
        "model": model,
        "k": k,
        "temperature": temperature,
        "run": run,
        "n": n,
        "em_voted": round(em_voted, 4),
        "f1_voted": round(f1_voted, 4),
        "oracle_k": round(oracle_k, 4),
        "agreement_avg": round(agree_avg, 4),
        "perfect_agreement_pct": round(perfect * 100, 1),
        "details": details,
    }


# ── 从已有 D2 结果复制第零次运行 ───────────────────────────────────

def seed_run0(models: list[str], k: int) -> None:
    """将已有的 D2 结果复制为第零次运行。"""
    os.makedirs(VARIANCE_DIR, exist_ok=True)
    for model in models:
        src = _original_path(model, k)
        dst = _run_path(model, k, 0)
        if os.path.exists(dst):
            continue
        if os.path.exists(src):
            shutil.copy2(src, dst)
            # 添加运行编号字段
            with open(dst) as f:
                data = json.load(f)
            data["run"] = 0
            with open(dst, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  run 0 ← {src}")
        else:
            print(f"  [WARN] Pas de résultat D2 pour {model}")


# ── 方差分析 ────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ci95(xs: list[float]) -> float:
    """计算 95% 置信区间的半宽度，使用 t 分布近似。"""
    if len(xs) < 2:
        return 0.0
    # t_{0.025} 对应 n-1 个自由度；n=3 到 10 时使用近似值
    t_vals = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    df = len(xs) - 1
    t = t_vals.get(df, 1.96)  # 找不到 t 值时使用 z=1.96
    return t * _std(xs) / math.sqrt(len(xs))


def analyze_variance(models: list[str], k: int, n_runs: int) -> dict:
    """分析多次运行之间的方差并返回摘要。"""
    print(f"\n{'='*80}")
    print(f"ANALYSE DE VARIANCE — {n_runs} runs, k={k}")
    print(f"{'='*80}\n")

    summary = {}
    for model in models:
        runs_data = []
        for r in range(n_runs):
            path = _run_path(model, k, r)
            if not os.path.exists(path):
                print(f"  [MANQUE] {model} run {r}")
                continue
            with open(path) as f:
                data = json.load(f)
            if data.get("n", 0) < 181:
                print(f"  [INCOMPLET] {model} run {r}: "
                      f"n={data.get('n', 0)}/181")
                continue
            runs_data.append(data)

        if len(runs_data) < 2:
            print(f"  {model}: seulement {len(runs_data)} run(s) complet(s), "
                  f"variance non calculable\n")
            continue

        metrics = {}
        for metric in ["em_voted", "f1_voted", "oracle_k",
                        "agreement_avg"]:
            vals = [d[metric] for d in runs_data]
            metrics[metric] = {
                "values": vals,
                "mean": round(_mean(vals), 4),
                "std": round(_std(vals), 4),
                "ci95": round(_ci95(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }

        # 逐题稳定性：有多少次运行给出相同的完全匹配结果？
        n_questions = runs_data[0]["n"]
        question_stability = []
        for qi in range(n_questions):
            ems = [d["details"][qi]["em_voted"] for d in runs_data
                   if qi < len(d.get("details", []))]
            # 与多数结果一致的运行所占比例
            if ems:
                correct_count = sum(ems)
                stability = max(correct_count, len(ems) - correct_count) / len(ems)
                question_stability.append(stability)

        baseline = load_baseline_results(model)
        em_base = baseline["em"] if baseline else None

        model_summary = {
            "model": model,
            "n_runs": len(runs_data),
            "n_questions": n_questions,
            "metrics": metrics,
            "question_stability_mean": round(_mean(question_stability), 4),
            "em_baseline_T0": em_base,
        }
        summary[model] = model_summary

        # 展示结果
        print(f"  {model} ({len(runs_data)} runs)")
        print(f"  {'─'*50}")
        if em_base is not None:
            print(f"  EM baseline T≈0:  {em_base:.4f}")
        for metric, stats in metrics.items():
            label = metric.replace("_", " ").title()
            print(f"  {label:<20} {stats['mean']:.4f} "
                  f"± {stats['std']:.4f}  "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]  "
                  f"CI95: ±{stats['ci95']:.4f}")
        print(f"  Question stability: {_mean(question_stability):.2%}")
        print()

    # 可直接用于 LaTeX 的汇总表
    print_latex_table(summary)

    # 保存
    out_path = os.path.join(VARIANCE_DIR, "variance_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé → {out_path}")

    return summary


def print_latex_table(summary: dict) -> None:
    """以 LaTeX 格式显示方差表。"""
    print("% --- Table variance inter-runs (copier dans l'article) ---")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Variance inter-runs de la self-consistency "
          r"($k=5$, $T=0.7$, $R$ runs indépendants).}")
    print(r"\label{tab:variance}")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(r"Modèle & $\text{EM}_{\text{SC}}$ & "
          r"$\text{Oracle}_k$ & Accord & Stabilité \\")
    print(r"\midrule")

    for model, s in summary.items():
        m = s["metrics"]
        short = model.split(":")[0]
        em = m["em_voted"]
        ora = m["oracle_k"]
        agr = m["agreement_avg"]
        stab = s["question_stability_mean"]
        print(f"  {short} & "
              f"${em['mean']:.3f} \\pm {em['std']:.3f}$ & "
              f"${ora['mean']:.3f} \\pm {ora['std']:.3f}$ & "
              f"${agr['mean']:.2f} \\pm {agr['std']:.2f}$ & "
              f"${stab:.2f}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── 运行时间估算 ────────────────────────────────────────────────────────

def estimate_time(models: list[str], k: int, n_runs: int,
                  n_questions: int = 181) -> None:
    """根据已有运行结果估算总耗时。"""
    existing = 0
    missing = 0
    for model in models:
        for r in range(n_runs):
            path = _run_path(model, k, r)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if data.get("n", 0) >= n_questions:
                    existing += 1
                    continue
            missing += 1

    calls_per_run = n_questions * k  # 181 × 5 = 905
    total_calls = missing * calls_per_run
    # 估计每次 Ollama 调用约需三秒
    secs = total_calls * 3
    hours = secs / 3600

    print(f"\n  Runs existants : {existing}/{n_runs * len(models)}")
    print(f"  Runs à faire   : {missing}")
    print(f"  Appels Ollama  : {total_calls:,} "
          f"({missing} runs × {calls_per_run} appels)")
    print(f"  Temps estimé   : ~{hours:.1f}h "
          f"(à ~3s/appel, séquentiel)")


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Variance inter-runs de la self-consistency D2")
    parser.add_argument("--runs", type=int, default=5,
                        help="Nombre total de runs (incluant run 0 = D2)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Modèles à évaluer")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Nombre de réponses par question (default: 5)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Température (default: 0.7)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyser les runs existants sans en lancer")
    parser.add_argument("--estimate", action="store_true",
                        help="Estimer le temps sans lancer")
    args = parser.parse_args()

    print(f"D2 Variance inter-runs — R={args.runs}, k={args.k}, T={args.temp}")
    print(f"Modèles : {args.models}")

    # 第零次运行复用已有的 D2 数据
    seed_run0(args.models, args.k)

    if args.estimate:
        estimate_time(args.models, args.k, args.runs)
        return

    if args.analyze_only:
        analyze_variance(args.models, args.k, args.runs)
        return

    # 只加载一次数据
    questions = load_difficult_questions()
    print(f"Questions difficiles : {len(questions)}")
    print("Chargement des faits de support…")
    facts_by_idx = load_supporting_facts()
    print(f"Faits chargés pour {len(facts_by_idx)} questions.")

    estimate_time(args.models, args.k, args.runs, len(questions))

    # 运行第 1 到 R-1 次；第零次已从原始 D2 结果复制
    for run in range(1, args.runs):
        for model in args.models:
            evaluate_run(model, questions, facts_by_idx,
                         k=args.k, temperature=args.temp, run=run)

    # 最终分析
    analyze_variance(args.models, args.k, args.runs)


if __name__ == "__main__":
    main()
