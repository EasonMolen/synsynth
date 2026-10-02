#!/usr/bin/env python3
"""
建议 B：零样本到轻量 QLoRA 的学习曲线。

分别使用 {10, 50, 200, 500, 1000, 3000} 个 Re-DocRED 或 HotpotQA
样本对 Qwen2.5-7B-Instruct 进行 QLoRA 微调，评估每个检查点，
绘制 F1/EM 随训练样本数变化的曲线。

用法：
    python learning_curve.py --task extraction
    python learning_curve.py --task multihop
    python learning_curve.py --task all
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time

# ── 按需将 scripts/ 加入 PYTHONPATH ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import (
    WORKSPACE_ROOT, RESULTS_DIR, logger, safe_path, RANDOM_SEED,
)
from qlora_finetune import (
    QLORA_DATA_DIR, QLORA_MODELS_DIR, HF_CACHE,
    DEFAULT_BASE_MODEL, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, MAX_SEQ_LENGTH, WARMUP_RATIO, WEIGHT_DECAY,
    _unload_ollama_models,
)

# ── 学习曲线采样点 ────────────────────────────────────────────────────
CURVE_POINTS = [10, 50, 200, 500, 1000, 3000]

# ── 输出目录 ──────────────────────────────────────────────────────────
LC_RESULTS_DIR = safe_path("results", "learning_curve")
os.makedirs(LC_RESULTS_DIR, exist_ok=True)


def _subsample_jsonl(src_path: str, n: int, seed: int = RANDOM_SEED) -> str:
    """创建包含 n 行的 JSONL 子样本文件，并返回路径。"""
    dst_path = os.path.join(QLORA_DATA_DIR, f"{os.path.basename(src_path).replace('.jsonl', '')}_n{n}.jsonl")
    if os.path.exists(dst_path):
        with open(dst_path) as f:
            existing = sum(1 for _ in f)
        if existing == n:
            logger.info("Sous-échantillon déjà prêt : %s (%d lignes)", dst_path, n)
            return dst_path

    with open(src_path) as f:
        all_lines = f.readlines()

    rng = random.Random(seed)
    sampled = rng.sample(all_lines, min(n, len(all_lines)))
    with open(dst_path, "w") as f:
        f.writelines(sampled)
    logger.info("Sous-échantillon créé : %s (%d lignes)", dst_path, len(sampled))
    return dst_path


def _train_at_n(task: str, n: int, base_model: str = DEFAULT_BASE_MODEL) -> str:
    """用 n 个样本训练 QLoRA，并返回适配器路径。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    adapter_dir = os.path.join(QLORA_MODELS_DIR, f"{task}_n{n}", "adapter")
    if os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Adaptateur déjà entraîné : %s", adapter_dir)
        return adapter_dir

    # 完整的原始数据
    full_jsonl = os.path.join(QLORA_DATA_DIR, f"{task}_train.jsonl")
    if not os.path.exists(full_jsonl):
        raise FileNotFoundError(f"Données manquantes : {full_jsonl}")

    # 抽取子样本
    sub_jsonl = _subsample_jsonl(full_jsonl, n)

    samples = []
    with open(sub_jsonl, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    dataset = Dataset.from_list(samples)

    logger.info("=" * 60)
    logger.info("Learning curve — tâche=%s, n_train=%d", task, n)
    logger.info("=" * 60)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4 位量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 将对话模板转换为纯文本
    def _apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    # 调整超参数：小样本（n <= 50）训练 10 个周期，否则至少 3 个
    num_epochs = 3 if n >= 200 else 10
    lr = 2e-4 if n <= 200 else 1e-4  # 大样本使用较低学习率，避免过拟合
    grad_accum = max(1, min(8, n // 2))  # 避免梯度累积步数超过每批样本数
    batch_size = min(2, n)

    ckpt_dir = os.path.join(QLORA_MODELS_DIR, f"{task}_n{n}", "checkpoints")
    training_args = SFTConfig(
        output_dir=ckpt_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=max(1, n // 20),
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        seed=RANDOM_SEED,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    t0 = time.time()
    logger.info("Entraînement n=%d (epochs=%d, lr=%.0e, batch=%d×%d)...",
                n, num_epochs, lr, batch_size, grad_accum)
    trainer.train()
    train_time = time.time() - t0
    logger.info("Entraînement terminé en %.0f s.", train_time)

    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(adapter_dir, "qlora_meta.json"), "w") as f:
        json.dump({
            "base_model": base_model,
            "task": task,
            "n_train": n,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "train_samples": len(dataset),
            "elapsed_seconds": round(train_time, 1),
        }, f, indent=2)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def _evaluate_extraction(adapter_dir: str, n_train: int) -> dict:
    """在 500 个 DocRED 样本上评估关系抽取适配器。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _unload_ollama_models()

    with open(os.path.join(adapter_dir, "qlora_meta.json")) as f:
        meta = json.load(f)
    base_model = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", cache_dir=HF_CACHE,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # 运行时替换 synsynth_model，使其使用 Hugging Face 模型
    import synsynth_model
    from qlora_finetune import (
        _hf_generate, _hf_generate_structured,
        TEMPERATURE, TOP_P,
    )
    import qlora_finetune
    qlora_finetune._qlora_model = model
    qlora_finetune._qlora_tokenizer = tokenizer

    orig_gen = synsynth_model.generate
    orig_gen_struct = synsynth_model.generate_structured
    synsynth_model.generate = _hf_generate
    synsynth_model.generate_structured = _hf_generate_structured

    try:
        from exp_extraction import run as run_extraction
        logger.info("Évaluation extraction (n_train=%d)...", n_train)
        result = run_extraction()
        result["n_train"] = n_train
        result["adapter_dir"] = adapter_dir
    finally:
        synsynth_model.generate = orig_gen
        synsynth_model.generate_structured = orig_gen_struct
        qlora_finetune._qlora_model = None
        qlora_finetune._qlora_tokenizer = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def _evaluate_multihop(adapter_dir: str, n_train: int) -> dict:
    """在 500 个 HotpotQA 样本上评估多跳推理适配器。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _unload_ollama_models()

    with open(os.path.join(adapter_dir, "qlora_meta.json")) as f:
        meta = json.load(f)
    base_model = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", cache_dir=HF_CACHE,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    import synsynth_model
    import qlora_finetune
    from qlora_finetune import _hf_generate, _hf_generate_structured
    qlora_finetune._qlora_model = model
    qlora_finetune._qlora_tokenizer = tokenizer

    orig_gen = synsynth_model.generate
    orig_gen_struct = synsynth_model.generate_structured
    synsynth_model.generate = _hf_generate
    synsynth_model.generate_structured = _hf_generate_structured

    try:
        from exp_multihop import run as run_multihop
        logger.info("Évaluation multihop (n_train=%d)...", n_train)
        result = run_multihop()
        result["n_train"] = n_train
        result["adapter_dir"] = adapter_dir
    finally:
        synsynth_model.generate = orig_gen
        synsynth_model.generate_structured = orig_gen_struct
        qlora_finetune._qlora_model = None
        qlora_finetune._qlora_tokenizer = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_learning_curve(task: str) -> list[dict]:
    """运行某项任务的完整学习曲线实验。"""
    results = []
    output_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.json")

    # 加载已有结果
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ns = {r["n_train"] for r in results}
        logger.info("Résultats existants pour %s : n_train=%s", task, sorted(done_ns))
    else:
        done_ns = set()

    evaluate_fn = _evaluate_extraction if task == "extraction" else _evaluate_multihop
    # multihop_v2 与 multihop 使用相同测试集和评估器

    for n in CURVE_POINTS:
        if n in done_ns:
            logger.info("Point n=%d déjà évalué — skip.", n)
            continue

        logger.info("━" * 60)
        logger.info("COURBE D'APPRENTISSAGE — %s — n_train=%d", task, n)
        logger.info("━" * 60)

        # 训练
        adapter_dir = _train_at_n(task, n)

        # 评估
        result = evaluate_fn(adapter_dir, n)
        results.append(result)

        # 逐步保存结果
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Résultat sauvegardé → %s", output_path)

    # 根据已有结果加入零样本数据点（n=0）
    if 0 not in {r.get("n_train") for r in results}:
        # multihop_v2 与 multihop 共用零样本基线
        zs_task = "multihop" if task in ("multihop_v2", "multihop_v3", "multihop_v4") else task
        zs_path = os.path.join(RESULTS_DIR, f"{zs_task}.json")
        if os.path.exists(zs_path):
            with open(zs_path) as f:
                zs = json.load(f)
            zs["n_train"] = 0
            zs["method"] = "zero-shot"
            results.insert(0, zs)

    # 按训练样本数排序
    results.sort(key=lambda r: r.get("n_train", 0))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Courbe d'apprentissage complète → %s", output_path)
    return results


def plot_learning_curve(task: str):
    """绘制学习曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lc_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.json")
    with open(lc_path) as f:
        results = json.load(f)

    ns = [r["n_train"] for r in results]
    if task == "extraction":
        scores = [r.get("f1_score", 0) for r in results]
        metric_label = "F1 Score"
    else:  # 处理 multihop 及其 V2、V3、V4 版本
        scores = [r.get("exact_match", r.get("accuracy", 0)) for r in results]
        metric_label = "Exact Match"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, scores, "o-", color="#2196F3", linewidth=2, markersize=8, label="QLoRA 4-bit (Qwen2.5-7B)")

    # Gemma-4-27B 的零样本基线
    if task == "extraction":
        ax.axhline(y=0.7023, color="#FF9800", linestyle="--", linewidth=1.5,
                    label="Zero-shot Gemma-4-27B (F1=0.70)")
        ax.axhline(y=0.802, color="#4CAF50", linestyle=":", linewidth=1.5,
                    label="DREEAM supervisé (F1=0.80)")
    elif task in ("multihop", "multihop_v2", "multihop_v3", "multihop_v4"):
        ax.axhline(y=0.462, color="#FF9800", linestyle="--", linewidth=1.5,
                    label="Zero-shot Phi-4-14B (EM=0.46)")

    ax.set_xlabel("Nombre d'exemples d'entraînement", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    task_label = task.replace("_v4", " V4").replace("_v3", " V3").replace("_v2", " V2").replace("_", " ").capitalize()
    ax.set_title(f"Courbe d'apprentissage — {task_label}\n"
                 f"QLoRA 4-bit sur RTX 3090 (24 Go)", fontsize=13)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Courbe d'apprentissage zero-shot → QLoRA")
    parser.add_argument("--task", choices=["extraction", "multihop", "multihop_v2", "multihop_v3", "multihop_v4", "all"], default="all")
    parser.add_argument("--plot-only", action="store_true",
                        help="Tracer les courbes sans relancer les entraînements.")
    args = parser.parse_args()

    tasks = ["extraction", "multihop"] if args.task == "all" else [args.task]  # multihop_v2 需明确指定

    for task in tasks:
        if not args.plot_only:
            run_learning_curve(task)
        plot_learning_curve(task)


if __name__ == "__main__":
    main()
