#!/usr/bin/env python3
"""
级联流程 V5b 的多次运行方差（Phi-4 → GPT-OSS，k=5）。

对完整级联流程（500 道题）独立运行 R 次，
测量完全匹配率、F1、Oracle 指标和重新路由比例的方差。

用法：
    python variance_cascade.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_self_consistency import (
    evaluate_pipeline_cascade, load_multihop_data, RESULTS_SC_DIR,
    answer_match, majority_vote, vote_agreement, bootstrap_ci,
)

CASCADE_MODELS = ["phi4:latest", "gpt-oss:20b"]
K = 5
TEMP = 0.7
THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = 0.4


def canonical_path():
    safe_p = CASCADE_MODELS[0].replace(":", "_").replace("/", "_")
    safe_f = CASCADE_MODELS[1].replace(":", "_").replace("/", "_")
    return os.path.join(RESULTS_SC_DIR, f"cascade_{safe_p}_{safe_f}_k{K}.json")


def run_path(run_id: int):
    safe_p = CASCADE_MODELS[0].replace(":", "_").replace("/", "_")
    safe_f = CASCADE_MODELS[1].replace(":", "_").replace("/", "_")
    return os.path.join(
        RESULTS_SC_DIR, f"cascade_{safe_p}_{safe_f}_k{K}_run{run_id}.json")


def run_cascade(data, run_id: int) -> dict:
    """运行一次级联实验，并以本次运行专用的文件名保存。"""
    canon = canonical_path()
    rp = run_path(run_id)

    # 如果本次运行已经完成，直接加载并返回结果
    if os.path.exists(rp):
        with open(rp) as f:
            existing = json.load(f)
        if len(existing.get("details", [])) >= len(data):
            print(f"\n  [SKIP] run {run_id} already complete ({rp})")
            return existing

    # 暂时移开已有的通用检查点
    canon_backup = None
    if os.path.exists(canon):
        canon_backup = canon + f".bak_variance"
        shutil.move(canon, canon_backup)

    # 如果存在本次运行的部分结果，将其复制到通用检查点路径
    if os.path.exists(rp):
        shutil.copy2(rp, canon)

    try:
        result = evaluate_pipeline_cascade(
            CASCADE_MODELS, data,
            k=K, temperature=TEMP,
            threshold_high=THRESHOLD_HIGH,
            threshold_low=THRESHOLD_LOW,
        )
    finally:
        # 以本次运行专用的文件名保存结果
        if os.path.exists(canon):
            shutil.copy2(canon, rp)
            os.remove(canon)
        # 如果原先存在通用检查点，将其恢复
        if canon_backup and os.path.exists(canon_backup):
            shutil.move(canon_backup, canon)

    return result


def compile_variance(runs_data: list[dict]) -> dict:
    """汇总 R 次运行的方差统计。"""
    R = len(runs_data)
    ems = [r["em_voted"] for r in runs_data]
    f1s = [r["f1_voted"] for r in runs_data]
    oracles = [r["oracle_k"] for r in runs_data]
    agrees = [r["agreement_avg"] for r in runs_data]
    pct_rerouted = [r["routing_stats"]["pct_rerouted"] for r in runs_data]

    print(f"\n{'='*70}")
    print(f"VARIANCE CASCADE V5b — R={R} runs")
    print(f"{'='*70}")
    print(f"{'Run':>4} {'EM':>7} {'F1':>7} {'Oracle':>7} {'Accord':>7} {'%rerou':>7}")
    print(f"{'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for i, r in enumerate(runs_data):
        print(f"{i:>4} {r['em_voted']:>7.4f} {r['f1_voted']:>7.4f} "
              f"{r['oracle_k']:>7.4f} {r['agreement_avg']:>7.4f} "
              f"{r['routing_stats']['pct_rerouted']:>6.1f}%")

    def stats_row(name, values):
        mu = statistics.mean(values)
        sigma = statistics.stdev(values) if len(values) > 1 else 0
        return f"  {name:<15} μ={mu:.4f}  σ={sigma:.4f}  [{min(values):.4f} ; {max(values):.4f}]"

    print(f"\n{'--- Résumé ---':^70}")
    print(stats_row("EM voté", ems))
    print(stats_row("F1 voté", f1s))
    print(stats_row("Oracle", oracles))
    print(stats_row("Accord moyen", agrees))
    print(stats_row("% rerouté", pct_rerouted))

    result = {
        "R": R,
        "models": CASCADE_MODELS,
        "k": K, "temperature": TEMP,
        "threshold_high": THRESHOLD_HIGH,
        "threshold_low": THRESHOLD_LOW,
        "em": {"mean": round(statistics.mean(ems), 4),
               "std": round(statistics.stdev(ems), 4) if R > 1 else 0,
               "values": [round(v, 4) for v in ems]},
        "f1": {"mean": round(statistics.mean(f1s), 4),
               "std": round(statistics.stdev(f1s), 4) if R > 1 else 0,
               "values": [round(v, 4) for v in f1s]},
        "oracle": {"mean": round(statistics.mean(oracles), 4),
                   "std": round(statistics.stdev(oracles), 4) if R > 1 else 0,
                   "values": [round(v, 4) for v in oracles]},
        "agreement": {"mean": round(statistics.mean(agrees), 4),
                      "std": round(statistics.stdev(agrees), 4) if R > 1 else 0,
                      "values": [round(v, 4) for v in agrees]},
        "pct_rerouted": {"mean": round(statistics.mean(pct_rerouted), 1),
                         "std": round(statistics.stdev(pct_rerouted), 1) if R > 1 else 0,
                         "values": [round(v, 1) for v in pct_rerouted]},
    }

    out_path = os.path.join(RESULTS_SC_DIR, "variance_cascade_v5b.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nRésultat → {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Variance inter-runs de la cascade V5b")
    parser.add_argument("--runs", "-R", type=int, default=3,
                        help="Nombre de runs (default: 3)")
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre de questions (default: toutes = 500)")
    parser.add_argument("--start-run", type=int, default=0,
                        help="Premier run à exécuter (default: 0)")
    args = parser.parse_args()

    data = load_multihop_data(args.n)
    print(f"Questions multi-hop : {len(data)}")
    print(f"Runs planifiés : {args.start_run} → {args.runs - 1}")

    # 将已有的级联结果视为第零次运行
    canon = canonical_path()
    rp0 = run_path(0)
    if os.path.exists(canon) and not os.path.exists(rp0):
        print(f"  Copie run original → run0")
        shutil.copy2(canon, rp0)

    runs_data = []
    for run_id in range(args.runs):
        print(f"\n{'#'*70}")
        print(f"# RUN {run_id}/{args.runs - 1}")
        print(f"{'#'*70}")

        if run_id < args.start_run:
            # 加载已有结果
            rp = run_path(run_id)
            if os.path.exists(rp):
                with open(rp) as f:
                    runs_data.append(json.load(f))
                print(f"  [LOADED] run {run_id}")
            else:
                print(f"  [ERROR] run {run_id} not found at {rp}")
                sys.exit(1)
        else:
            t0 = time.time()
            result = run_cascade(data, run_id)
            elapsed = time.time() - t0
            runs_data.append(result)
            print(f"  Run {run_id} terminé en {elapsed/3600:.1f}h "
                  f"(EM={result['em_voted']:.4f})")

    compile_variance(runs_data)


if __name__ == "__main__":
    main()
