#!/usr/bin/env python3
"""
SYNSYNTH+ 的多模型基准测试。

使用每项任务的小批量样本评估各 Ollama 模型：
  1. 关系抽取（JSON 解析与 F1）
  2. 文本到查询（生成 Cypher 并评估准确率）
  3. 多跳推理（完全匹配）
  4. RAG 忠实度

用法：
    python benchmark_models.py [--samples N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ── 将 scripts/ 加入模块搜索路径，再导入项目模块 ─────────────────────────
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

from synsynth_config import logger, RESULTS_DIR

# ── 各任务的候选模型 ──────────────────────────────────────────────────
# （模型名称，选择理由）
EXTRACTION_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "phi4:latest",
    "mistral-small:latest",
    "nuextract:latest",
    "llama3.1:8b",
]

QUERY_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "qwen2.5-coder:32b",
    "phi4:latest",
    "mistral-small:latest",
]

MULTIHOP_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "deepseek-r1:32b",
    "phi4:latest",
    "mistral-small:latest",
]

RAG_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "command-r7b:latest",
    "phi4:latest",
    "mistral-small:latest",
    "llama3.1:8b",
]


def set_model(model_name: str):
    """通过运行时替换设置 synsynth_model 中的当前模型。"""
    import synsynth_model
    synsynth_model.OLLAMA_MODEL = model_name
    logger.info("── Modèle changé → %s ──", model_name)


def warm_up(model_name: str) -> bool:
    """用简单提示词预热模型；成功时返回 True。"""
    import synsynth_model
    try:
        set_model(model_name)
        resp = synsynth_model.generate("Réponds 'ok'.", max_new_tokens=8)
        logger.info("  warm-up OK: %s", resp[:50])
        return True
    except Exception as e:
        logger.error("  warm-up ÉCHEC pour %s: %s", model_name, e)
        return False


# ── 关系抽取基准测试 ──────────────────────────────────────────────────
def bench_extraction(n_samples: int) -> dict:
    """使用当前模型评估 n_samples 个关系抽取样本。"""
    import exp_extraction
    return exp_extraction.run(n_samples)


# ── 查询生成基准测试 ──────────────────────────────────────────────────
def bench_query(n_samples: int) -> dict:
    """使用当前模型评估 n_samples 个文本到查询样本。"""
    import exp_query
    return exp_query.run(n_samples)


# ── 多跳推理基准测试 ──────────────────────────────────────────────────
def bench_multihop(n_samples: int) -> dict:
    """评估 n_samples 个多跳推理样本。"""
    import exp_multihop
    return exp_multihop.run(n_samples)


# ── RAG 基准测试 ──────────────────────────────────────────────────────
def bench_rag(n_samples: int) -> dict:
    """评估 n_samples 个 RAG 忠实度样本。"""
    import exp_rag
    return exp_rag.run(n_samples)


# ── 辅助函数 ──────────────────────────────────────────────────────────
def extract_metrics(result: dict, task: str) -> dict:
    """从实验结果中提取主要指标。"""
    m = {"task": task}
    if task == "extraction":
        m["f1"] = result.get("f1_score", 0)
        m["precision"] = result.get("precision", 0)
        m["recall"] = result.get("recall", 0)
        # 计算解析失败率
        details = result.get("details", [])
        pf = sum(1 for d in details if d.get("status") == "parse_fail")
        m["parse_fail_rate"] = round(pf / len(details), 4) if details else 0
    elif task == "query":
        m["accuracy"] = result.get("accuracy", 0)
        m["cypher_valid"] = result.get("cypher_syntax_valid_rate", 0)
    elif task == "multihop":
        m["exact_match"] = result.get("exact_match_accuracy", 0)
        # 如果缺少汇总值，则根据明细计算
        if m["exact_match"] == 0:
            details = result.get("details", [])
            if details:
                em = sum(1 for d in details if d.get("exact_match"))
                m["exact_match"] = round(em / len(details), 4)
    elif task == "rag":
        m["faithfulness"] = result.get("avg_faithfulness", 0)
        m["relevance"] = result.get("avg_answer_relevance", 0)
        m["context_precision"] = result.get("avg_context_precision", 0)
    m["elapsed_s"] = result.get("elapsed_seconds", 0)
    m["n_samples"] = result.get("n_samples", 0)
    return m


def run_task_benchmark(task: str, models: list[str], n_samples: int,
                       bench_fn) -> list[dict]:
    """对某任务的全部候选模型运行基准测试。"""
    results = []
    for model in models:
        logger.info("=" * 60)
        logger.info("BENCHMARK %s — modèle: %s — %d samples", task.upper(), model, n_samples)
        logger.info("=" * 60)

        if not warm_up(model):
            results.append({
                "model": model, "task": task, "status": "error",
                "error": "warm-up failed"
            })
            continue

        try:
            t0 = time.time()
            raw_result = bench_fn(n_samples)
            elapsed = time.time() - t0

            metrics = extract_metrics(raw_result, task)
            metrics["model"] = model
            metrics["status"] = "ok"
            results.append(metrics)

            logger.info("RÉSULTAT %s / %s: %s [%.1fs]",
                        task, model, json.dumps(metrics, ensure_ascii=False), elapsed)
        except Exception as e:
            logger.error("ERREUR %s / %s: %s", task, model, e)
            results.append({
                "model": model, "task": task, "status": "error",
                "error": str(e)
            })

    return results


def print_leaderboard(all_results: list[dict]):
    """按任务显示汇总表。"""
    tasks = {}
    for r in all_results:
        task = r.get("task", "?")
        tasks.setdefault(task, []).append(r)

    print("\n" + "=" * 80)
    print("LEADERBOARD — Benchmark multi-modèles SYNSYNTH+")
    print("=" * 80)

    best_per_task = {}

    for task, results in tasks.items():
        print(f"\n{'─' * 60}")
        print(f"  TÂCHE: {task.upper()}")
        print(f"{'─' * 60}")

        # 按主要指标排序
        if task == "extraction":
            key = "f1"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'PFail%':>7} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('f1', 0):>6.3f} {r.get('precision', 0):>6.3f} "
                          f"{r.get('recall', 0):>6.3f} {r.get('parse_fail_rate', 0)*100:>6.1f}% "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "query":
            key = "accuracy"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'Acc':>6} {'Cypher':>7} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('accuracy', 0):>6.3f} "
                          f"{r.get('cypher_valid', 0):>7.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "multihop":
            key = "exact_match"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'EM':>6} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('exact_match', 0):>6.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "rag":
            key = "faithfulness"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'Faith':>6} {'Relev':>6} {'CtxP':>6} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('faithfulness', 0):>6.3f} "
                          f"{r.get('relevance', 0):>6.3f} "
                          f"{r.get('context_precision', 0):>6.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        else:
            key = None

        # 最佳模型
        ok_results = [r for r in results if r.get("status") == "ok"]
        if ok_results and key:
            best = ok_results[0]
            best_per_task[task] = best["model"]
            print(f"\n  ★ MEILLEUR: {best['model']} ({key}={best.get(key, 0):.3f})")

    print(f"\n{'=' * 80}")
    print("RÉSUMÉ — Meilleur modèle par tâche :")
    for task, model in best_per_task.items():
        print(f"  • {task:15s} → {model}")
    print("=" * 80)

    return best_per_task


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-modèles SYNSYNTH+")
    parser.add_argument("--samples", "-n", type=int, default=5,
                        help="Nombre d'échantillons par tâche (défaut: 5)")
    parser.add_argument("--tasks", "-t", nargs="+",
                        choices=["extraction", "query", "multihop", "rag", "all"],
                        default=["all"],
                        help="Tâches à benchmarker (défaut: all)")
    args = parser.parse_args()

    n = args.samples
    do_all = "all" in args.tasks

    all_results = []
    t_global = time.time()

    if do_all or "extraction" in args.tasks:
        all_results.extend(
            run_task_benchmark("extraction", EXTRACTION_MODELS, n, bench_extraction))

    if do_all or "query" in args.tasks:
        all_results.extend(
            run_task_benchmark("query", QUERY_MODELS, n, bench_query))

    if do_all or "multihop" in args.tasks:
        all_results.extend(
            run_task_benchmark("multihop", MULTIHOP_MODELS, n, bench_multihop))

    if do_all or "rag" in args.tasks:
        all_results.extend(
            run_task_benchmark("rag", RAG_MODELS, n, bench_rag))

    total_time = time.time() - t_global

    # 显示排行榜
    best_per_task = print_leaderboard(all_results)

    # 保存结果
    out_file = os.path.join(RESULTS_DIR, "benchmark_models.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "samples_per_task": n,
            "total_elapsed_seconds": round(total_time, 1),
            "best_per_task": best_per_task,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("Résultats sauvegardés dans %s", out_file)
    logger.info("Temps total: %.1f min", total_time / 60)


if __name__ == "__main__":
    main()
