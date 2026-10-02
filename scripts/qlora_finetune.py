#!/usr/bin/env python3
"""
SYNSYNTH+ 的 QLoRA 微调模块（阶段 2b）。

使用监督数据和 4 位 QLoRA 微调大语言模型
（默认 Qwen2.5-7B-Instruct），作为零样本流程的性能参照。

支持的任务：
  - extraction：Re-DocRED 关系抽取
  - multihop：HotpotQA 多跳推理

独立运行：
    python qlora_finetune.py --task extraction
    python qlora_finetune.py --task multihop
    python qlora_finetune.py --task all

通过主流程运行：
    python run_synsynth.py --qlora --gbnf --seed 42
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time
import logging

from synsynth_config import (
    WORKSPACE_ROOT, MODELS_DIR, DATA_DIR, CACHE_DIR,
    RANDOM_SEED, TEMPERATURE, TOP_P, logger, safe_path,
)

# ── 支持 QLoRA 的任务 ────────────────────────────────────────────────
QLORA_TASKS = ["extraction", "multihop"]

# ── 基础模型，公开可用，无需 Hugging Face 身份验证 ──────────────────
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

QLORA_BASE_MODELS: dict[str, str] = {
    "extraction": DEFAULT_BASE_MODEL,
    "multihop":   DEFAULT_BASE_MODEL,
}

# ── LoRA 超参数 ───────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── 训练超参数 ────────────────────────────────────────────────────────
LEARNING_RATE         = 2e-4
NUM_EPOCHS            = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8        # 有效批量大小为 16
MAX_SEQ_LENGTH        = 2048
WARMUP_RATIO          = 0.03
WEIGHT_DECAY          = 0.01
MAX_TRAIN_SAMPLES     = 3000

# ── 路径 ──────────────────────────────────────────────────────────────
QLORA_DATA_DIR   = safe_path("data", "qlora")
QLORA_MODELS_DIR = safe_path("models", "qlora")
HF_CACHE         = safe_path("cache", "huggingface")

for _d in (QLORA_DATA_DIR, QLORA_MODELS_DIR, HF_CACHE):
    os.makedirs(_d, exist_ok=True)


# ====================================================================
#  准备数据（调用外部 PJKG5 项目）
# ====================================================================

def prepare_qlora_data(task: str) -> str:
    """准备 JSONL 训练数据并返回文件路径。"""
    pjkg5 = os.path.join(os.path.dirname(WORKSPACE_ROOT), "PJKG5")
    data_module_path = os.path.join(pjkg5, "prepare_qlora_data.py")
    if not os.path.isfile(data_module_path):
        raise FileNotFoundError(
            f"QLoRA 数据准备需要外部文件 {data_module_path}。"
        )
    if pjkg5 not in sys.path:
        sys.path.insert(0, pjkg5)

    prepare_data = importlib.import_module("prepare_qlora_data")

    output_path = os.path.join(QLORA_DATA_DIR, f"{task}_train.jsonl")

    # 如果已有文件中的样本足够，则直接复用
    if os.path.exists(output_path):
        with open(output_path) as f:
            n = sum(1 for _ in f)
        if n >= MAX_TRAIN_SAMPLES * 0.9:
            logger.info("Données QLoRA '%s' déjà prêtes (%d samples).", task, n)
            return output_path

    if task == "extraction":
        samples = prepare_data.load_redocred_for_qlora(MAX_TRAIN_SAMPLES, HF_CACHE)
    elif task == "multihop":
        samples = prepare_data.load_hotpotqa_for_qlora(MAX_TRAIN_SAMPLES, HF_CACHE)
    else:
        raise ValueError(f"QLoRA non supporté pour la tâche : {task}")

    prepare_data.write_jsonl(samples, output_path)
    return output_path


# ====================================================================
#  QLoRA 训练
# ====================================================================

def has_finetuned_model(task: str) -> bool:
    """检查该任务是否已有微调后的适配器。"""
    adapter_dir = os.path.join(QLORA_MODELS_DIR, task, "adapter")
    return os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))


def finetune(
    task: str,
    base_model: str | None = None,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> str:
    """使用 QLoRA 微调并返回适配器目录。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    if base_model is None:
        base_model = QLORA_BASE_MODELS.get(task, DEFAULT_BASE_MODEL)

    adapter_dir = os.path.join(QLORA_MODELS_DIR, task, "adapter")

    if has_finetuned_model(task):
        logger.info("Adaptateur QLoRA '%s' déjà entraîné → %s", task, adapter_dir)
        return adapter_dir

    logger.info("=" * 60)
    logger.info("QLoRA fine-tuning : tâche=%s, modèle=%s", task, base_model)
    logger.info("=" * 60)

    # 1. 加载数据
    data_path = prepare_qlora_data(task)
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    dataset = Dataset.from_list(samples)
    logger.info("Données : %d samples depuis %s", len(dataset), data_path)

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 进行 4 位量化（NF4）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. 加载基础模型
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

    # 5. 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 预处理数据集：将对话模板转换为纯文本
    def _apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    # 7. 配置监督微调
    ckpt_dir = os.path.join(QLORA_MODELS_DIR, task, "checkpoints")
    training_args = SFTConfig(
        output_dir=ckpt_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=learning_rate,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=RANDOM_SEED,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 8. 创建训练器
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    # 9. 开始训练
    t0 = time.time()
    logger.info("Début de l'entraînement QLoRA (%d epochs, lr=%.0e)...", num_epochs, learning_rate)
    trainer.train()
    elapsed = time.time() - t0
    logger.info("Entraînement terminé en %.0f s.", elapsed)

    # 10. 保存适配器和元数据
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(adapter_dir, "qlora_meta.json"), "w") as f:
        json.dump({
            "base_model": base_model,
            "task": task,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "train_samples": len(dataset),
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    logger.info("Adaptateur sauvegardé → %s", adapter_dir)

    # 释放显存
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir


# ====================================================================
#  使用 Hugging Face 推理（运行时替换 synsynth_model）
# ====================================================================

_qlora_model = None
_qlora_tokenizer = None
_original_generate = None
_original_generate_structured = None


def _unload_ollama_models():
    """从显存卸载所有 Ollama 模型，为 GPU 腾出空间。"""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/ps")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            logger.info("Aucun modèle Ollama en VRAM.")
            return
        for m in models:
            name = m.get("name", m.get("model", ""))
            if not name:
                continue
            logger.info("Déchargement Ollama : %s", name)
            payload = json.dumps({"model": name, "keep_alive": 0}).encode()
            req2 = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req2, timeout=30) as resp2:
                    resp2.read()
            except Exception as e:
                logger.warning("Erreur déchargement '%s' : %s", name, e)
        logger.info("Modèles Ollama déchargés de la VRAM.")
    except Exception as e:
        logger.warning("Impossible de contacter Ollama pour décharger : %s", e)


def load_finetuned_model(task: str):
    """加载基础模型及 QLoRA 适配器，用于推理。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # 加载 QLoRA 模型前先释放 Ollama 占用的显存
    _unload_ollama_models()

    adapter_dir = os.path.join(QLORA_MODELS_DIR, task, "adapter")
    with open(os.path.join(adapter_dir, "qlora_meta.json")) as f:
        meta = json.load(f)
    base_model = meta["base_model"]

    logger.info("Chargement QLoRA '%s' (base=%s)...", task, base_model)

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
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    logger.info("Modèle QLoRA '%s' chargé.", task)
    return model, tokenizer


def _hf_generate(
    prompt: str,
    *,
    system: str = "",
    max_new_tokens: int = 2048,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    json_format: bool = False,
) -> str:
    """使用已加载的 Hugging Face QLoRA 模型生成回答。"""
    import torch

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    encoded = _qlora_tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"]
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    input_ids = input_ids.to("cuda")

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=_qlora_tokenizer.pad_token_id,
    )
    if temperature > 0.01:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = _qlora_model.generate(input_ids, **gen_kwargs)

    input_len = input_ids.shape[-1]
    generated = output[0][input_len:]
    text = _qlora_tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def _hf_generate_structured(
    prompt: str,
    *,
    system: str = "",
    json_mode: bool = False,
    max_new_tokens: int = 4096,
) -> str:
    """使用 Hugging Face QLoRA 模型生成结构化回答。"""
    if json_mode:
        system = (system + "\n" if system else "") + (
            "Tu dois répondre UNIQUEMENT avec un objet JSON valide, "
            "sans texte avant ni après."
        )
    return _hf_generate(
        prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        json_format=json_mode,
    )


def patch_inference(task: str):
    """运行时替换 synsynth_model 的推理函数，改用 QLoRA 模型。"""
    global _qlora_model, _qlora_tokenizer
    global _original_generate, _original_generate_structured

    _qlora_model, _qlora_tokenizer = load_finetuned_model(task)

    import synsynth_model
    _original_generate = synsynth_model.generate
    _original_generate_structured = synsynth_model.generate_structured

    synsynth_model.generate = _hf_generate
    synsynth_model.generate_structured = _hf_generate_structured

    logger.info("Inférence HuggingFace activée pour '%s'.", task)


def unpatch_inference():
    """恢复 Ollama 推理函数并释放显存。"""
    global _qlora_model, _qlora_tokenizer
    global _original_generate, _original_generate_structured

    if _original_generate is not None:
        import synsynth_model
        synsynth_model.generate = _original_generate
        synsynth_model.generate_structured = _original_generate_structured
        _original_generate = None
        _original_generate_structured = None
        logger.info("Inférence Ollama restaurée.")

    if _qlora_model is not None:
        del _qlora_model
        _qlora_model = None
    if _qlora_tokenizer is not None:
        del _qlora_tokenizer
        _qlora_tokenizer = None

    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ====================================================================
#  命令行入口（独立运行）
# ====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning pour SYNSYNTH+",
    )
    parser.add_argument(
        "--task", choices=["extraction", "multihop", "all"], default="all",
        help="Tâche à fine-tuner (défaut : all).",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help=f"Modèle de base HuggingFace (défaut : {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Nombre d'époques (défaut : {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (défaut : {LEARNING_RATE}).",
    )
    args = parser.parse_args()

    tasks = QLORA_TASKS if args.task == "all" else [args.task]

    for task in tasks:
        finetune(
            task,
            base_model=args.base_model,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )

    logger.info("Fine-tuning QLoRA terminé.")


if __name__ == "__main__":
    main()
