#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                   SYNSYNTH+ — 实验流程                               ║
║                                                                      ║
║  模型    : unsloth/gemma-4-26B-A4B-it-GGUF  (UD-Q4_K_XK)          ║
║  目标    : 四项实验评估与论文生成                                      ║
║  工作区  : 所有文件读写都限制在本项目目录中                            ║
╚══════════════════════════════════════════════════════════════════════╝

用法：
    python run_synsynth.py                   # 运行完整流程
    python run_synsynth.py --self-improve    # 完整流程加自改进循环（需要缺失的外部模块）
    python run_synsynth.py --exp extraction  # 只运行一个实验
    python run_synsynth.py --article-only    # 根据已有结果生成文章
    python run_synsynth.py --n-samples 20    # 减少样本数，快速测试
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ── 安全：确保当前工作目录位于项目内 ───────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(_SCRIPT_DIR)  # 项目目录是 scripts/ 的上一级
os.chdir(WORKSPACE)

# 将 scripts/ 加入模块搜索路径，供项目内部导入使用
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 将缓存目录限制在项目内
os.environ["HF_HOME"] = os.path.join(WORKSPACE, "cache", "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WORKSPACE, "cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(WORKSPACE, "cache", "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(WORKSPACE, "cache")

from synsynth_config import (
    logger, RESULTS_DIR, safe_path, DATA_DIR, ARTICLE_DIR,
    TASK_MODELS, DEFAULT_MODEL,
)
from synsynth_io import write_json, read_json, write_text
import synsynth_model

# 断点续跑模式的全局标记
_RESUME_MODE = False


# ============================================================================
#  流程调度函数
# ============================================================================

def run_experiment(name: str, n_samples: int | None = None) -> dict:
    """根据名称动态导入并运行实验。"""
    EXPERIMENTS = {
        "extraction":  "exp_extraction",
        "query":       "exp_query",
        "multihop":    "exp_multihop",
        "rag":         "exp_rag",
    }
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验：{name!r}。可选值：{list(EXPERIMENTS)}")

    # 为当前任务选择配置的模型
    model = TASK_MODELS.get(name, DEFAULT_MODEL)
    synsynth_model.OLLAMA_MODEL = model
    logger.info("实验 '%s' 使用模型：%s", name, model)

    mod = __import__(EXPERIMENTS[name])
    return mod.run(n_samples=n_samples)


def run_all_experiments(n_samples: int | None = None) -> dict[str, dict]:
    """依次运行四项实验。

    使用 --resume 时，跳过已有结果文件的实验。
    """
    results = {}
    exp_names = ["extraction", "query", "multihop", "rag"]
    exp_result_keys = {
        "extraction": "extraction",
        "query": "text_to_query",
        "multihop": "multihop_reasoning",
        "rag": "rag_faithfulness",
    }

    for name in exp_names:
        logger.info("━" * 60)

        # 续跑时检查是否已有完整结果
        if _RESUME_MODE:
            result_key = exp_result_keys[name]
            result_path = os.path.join(RESULTS_DIR, f"{result_key}.json")
            if os.path.exists(result_path):
                existing = read_json(f"results/{result_key}.json")
                if existing and "error" not in existing:
                    logger.info("⏭  实验 '%s' 已有结果，跳过。", name)
                    results[result_key] = existing
                    continue

        try:
            res = run_experiment(name, n_samples=n_samples)
            results[res.get("experiment", name)] = res
            # 每完成一项实验就保存结果
            write_json(
                f"results/{res.get('experiment', name)}.json",
                res,
            )
        except Exception as e:
            logger.error("实验 '%s' 失败：%s", name, e, exc_info=True)
            results[name] = {"experiment": name, "error": str(e)}

    return results


def generate_article(all_results: dict) -> str:
    """调用文章生成模块。"""
    from synsynth_article import generate_article as _gen
    return _gen(all_results)


def generate_visualizations(all_results: dict) -> list[str]:
    """调用可视化模块。"""
    try:
        from synsynth_viz import plot_summary
        return plot_summary(all_results)
    except ImportError as e:
        logger.warning("无法生成图表（可能缺少 matplotlib）：%s", e)
        return []


def load_existing_results() -> dict:
    """加载之前保存的结果。"""
    p = os.path.join(RESULTS_DIR, "all_results.json")
    if os.path.exists(p):
        rel = os.path.relpath(p, WORKSPACE)
        return read_json(rel)
    # 加载各实验单独保存的文件
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json") and fname != "all_results.json":
            rel = os.path.relpath(os.path.join(RESULTS_DIR, fname), WORKSPACE)
            data = read_json(rel)
            key = data.get("experiment", fname.replace(".json", ""))
            results[key] = data
    return results


# ============================================================================
#  程序入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SYNSYNTH+：基于本地模型的实验流程",
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        choices=["extraction", "query", "multihop", "rag"],
        help="只运行指定实验；默认运行全部实验。",
    )
    parser.add_argument(
        "--article-only", action="store_true",
        help="根据已有实验结果生成文章。",
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="只生成可视化图表。",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="每项实验的样本数；默认使用配置值。",
    )
    parser.add_argument(
        "--skip-article", action="store_true",
        help="运行实验，但不生成文章。",
    )
    parser.add_argument(
        "--self-improve", action="store_true",
        help="启用自改进循环；需要仓库中缺失的 synsynth_selfimprove.py。",
    )
    parser.add_argument(
        "--max-improve-iter", type=int, default=3,
        help="每项实验最多执行的自改进次数；默认 3 次。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="继续上次运行：跳过已完成的实验，并读取检查点。",
    )
    parser.add_argument(
        "--gbnf", action="store_true",
        help="启用 GBNF 约束解码；需要外部 gbnf_patch.py（默认位于相邻的 PJKG5 目录）。",
    )
    parser.add_argument(
        "--gbnf-patch-dir", type=str, default=None,
        help="gbnf_patch.py 所在目录；仅与 --gbnf 一起使用。",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子；默认 42。可用于测量不同运行之间的方差。",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="所有任务都使用指定的 Ollama 模型，例如 llama3.1:8b；用于基线比较。",
    )
    parser.add_argument(
        "--qlora", action="store_true",
        help="启用 QLoRA 微调（阶段 2b）：在 Re-DocRED/HotpotQA 上训练，"
             "通过 Hugging Face 推理评估，结果名称以 _qlora 结尾。",
    )
    parser.add_argument(
        "--qlora-base-model", type=str, default=None,
        help="QLoRA 的 Hugging Face 基础模型；默认 Qwen/Qwen2.5-7B-Instruct。",
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              SYNSYNTH+ 实验流程启动                      ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("工作区：%s", WORKSPACE)

    # ── 随机种子 ──────────────────────────────────────────────────────
    if args.seed is not None:
        import synsynth_config
        synsynth_config.RANDOM_SEED = args.seed
        import random
        random.seed(args.seed)
        logger.info("随机种子设为 %d", args.seed)

    # ── 为基线实验强制指定模型 ───────────────────────────────────────
    if args.model:
        for key in TASK_MODELS:
            TASK_MODELS[key] = args.model
        logger.info("全部任务使用模型：%s", args.model)

    # 按需启用断点续跑
    global _RESUME_MODE
    if args.resume:
        _RESUME_MODE = True
        logger.info("已启用断点续跑；跳过已完成实验。")

    # 按需启用 GBNF 约束解码；补丁来自外部项目
    if args.gbnf:
        patch_dir = os.path.abspath(
            args.gbnf_patch_dir or os.path.join(os.path.dirname(WORKSPACE), "PJKG5")
        )
        patch_file = os.path.join(patch_dir, "gbnf_patch.py")
        if not os.path.isfile(patch_file):
            parser.error(
                f"--gbnf 需要外部文件 {patch_file}。"
                "请提供 --gbnf-patch-dir，或不使用 --gbnf。"
            )
        if patch_dir not in sys.path:
            sys.path.insert(0, patch_dir)
        gbnf_patch = importlib.import_module("gbnf_patch")
        patch_model_module = gbnf_patch.patch_model_module
        set_task = gbnf_patch.set_task
        patch_model_module()
        # 包装实验函数，在每次实验前设置 GBNF 任务
        _original_run_experiment = run_experiment
        def _gbnf_run_experiment(name, n_samples=None):
            set_task(name)
            return _original_run_experiment(name, n_samples=n_samples)
        globals()['run_experiment'] = _gbnf_run_experiment
        logger.info("已启用 GBNF 约束解码（严格 JSON Schema）。")

    # 按需启用 QLoRA
    if args.qlora:
        from qlora_finetune import (
            finetune, has_finetuned_model, patch_inference, unpatch_inference,
            QLORA_TASKS,
        )

        # 1. 如有需要，训练模型
        for _task in QLORA_TASKS:
            if not has_finetuned_model(_task):
                logger.info("正在为 '%s' 训练 QLoRA...", _task)
                finetune(_task, base_model=args.qlora_base_model)

        # 2. 包装实验函数，按任务切换推理实现
        _prev_run_experiment_qlora = globals()['run_experiment']

        def _qlora_run_experiment(name, n_samples=None):
            if name in QLORA_TASKS and has_finetuned_model(name):
                patch_inference(name)
                try:
                    res = _prev_run_experiment_qlora(name, n_samples=n_samples)
                finally:
                    unpatch_inference()
                # 标记 QLoRA 实验结果
                original_exp = res.get("experiment", name)
                res["experiment"] = original_exp + "_qlora"
                res["method"] = "qlora"
                res["base_experiment"] = original_exp
                return res
            else:
                return _prev_run_experiment_qlora(name, n_samples=n_samples)

        globals()['run_experiment'] = _qlora_run_experiment
        logger.info("已为以下任务启用 QLoRA：%s", QLORA_TASKS)

    t_global = time.time()

    # ── 仅生成文章 ────────────────────────────────────────────────────
    if args.article_only:
        all_results = load_existing_results()
        if not all_results:
            logger.error("找不到实验结果，请先运行实验。")
            sys.exit(1)
        generate_article(all_results)
        logger.info("文章已生成 → article/SYNSYNTH_article.md")
        return

    # ── 仅生成图表 ────────────────────────────────────────────────────
    if args.viz_only:
        all_results = load_existing_results()
        if not all_results:
            logger.error("找不到实验结果。")
            sys.exit(1)
        paths = generate_visualizations(all_results)
        for p in paths:
            logger.info("图表 → %s", p)
        return

    # ── 单项实验 ──────────────────────────────────────────────────────
    if args.exp:
        if args.self_improve:
            if not os.path.isfile(os.path.join(_SCRIPT_DIR, "synsynth_selfimprove.py")):
                parser.error("--self-improve 需要仓库中缺失的 synsynth_selfimprove.py。")
            self_improve = importlib.import_module("synsynth_selfimprove").self_improve
            logger.info("实验 '%s' 已启用自改进（最多 %d 次）。",
                        args.exp, args.max_improve_iter)
            res = self_improve(
                args.exp,
                run_experiment,
                n_samples=args.n_samples,
                max_iterations=args.max_improve_iter,
            )
        else:
            res = run_experiment(args.exp, n_samples=args.n_samples)
        write_json(f"results/{res.get('experiment', args.exp)}.json", res)
        if not args.skip_article:
            all_results = load_existing_results()
            all_results[res.get("experiment", args.exp)] = res
            generate_article(all_results)
        return

    # ── 完整实验流程 ──────────────────────────────────────────────────
    if args.self_improve:
        if not os.path.isfile(os.path.join(_SCRIPT_DIR, "synsynth_selfimprove.py")):
            parser.error("--self-improve 需要仓库中缺失的 synsynth_selfimprove.py。")
        self_improve_all = importlib.import_module("synsynth_selfimprove").self_improve_all
        logger.info("已启用自改进（每项实验最多 %d 次）。",
                    args.max_improve_iter)
        all_results = self_improve_all(
            run_experiment,
            n_samples=args.n_samples,
            max_iterations=args.max_improve_iter,
        )
    else:
        all_results = run_all_experiments(n_samples=args.n_samples)

    # 保存全部实验结果
    write_json("results/all_results.json", all_results)

    # 生成图表
    fig_paths = generate_visualizations(all_results)

    # 生成文章
    if not args.skip_article:
        article = generate_article(all_results)
        logger.info("文章已保存 → article/SYNSYNTH_article.md")

    # ── 最终报告 ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_global
    logger.info("━" * 60)
    logger.info("流程耗时 %.1f 秒。", elapsed_total)
    logger.info("实验结果       → results/")
    logger.info("图表           → %s", ", ".join(fig_paths) if fig_paths else "（无）")
    logger.info("文章           → article/SYNSYNTH_article.md")
    logger.info("日志           → logs/synsynth.log")

    # 汇总各实验指标
    for key, res in all_results.items():
        if "error" in res:
            logger.warning("  %-25s  错误：%s", key, res["error"])
        else:
            score = (
                res.get("f1_score")
                or res.get("accuracy")
                or res.get("exact_accuracy")
                or res.get("avg_faithfulness")
                or res.get("avg_token_f1")
                or "?"
            )
            logger.info("  %-25s  得分 = %s", key, score)


if __name__ == "__main__":
    main()
