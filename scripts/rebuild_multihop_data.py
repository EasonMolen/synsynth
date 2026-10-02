#!/usr/bin/env python3
"""
利用 HotpotQA 的标准支持事实，重建包含推理链的多跳训练数据。

修复的问题：
    V1 数据中的推理链全部退化为：
        {"reasoning_chain": ["D'après les faits fournis"], "answer": "..."}
    生成这些数据的 Phi-4 模型没有输出真正的推理过程。

解决方法：
    根据 HotpotQA 的标准支持事实，用模板构建两到三步推理链，
    无需调用大语言模型。

用法：
    python scripts/rebuild_multihop_data.py
    python scripts/rebuild_multihop_data.py --max-samples 5000
"""
from __future__ import annotations

import json
import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import WORKSPACE_ROOT, logger, RANDOM_SEED

HF_CACHE = os.path.join(WORKSPACE_ROOT, "data", "hf_cache")
QLORA_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "qlora")

MULTIHOP_SYSTEM = (
    "Tu es un agent de raisonnement multi-hop. "
    "Tu reçois une question complexe et des faits de support. "
    "Tu dois raisonner étape par étape en reliant les faits, puis fournir "
    "ta réponse finale COURTE et PRÉCISE (quelques mots seulement). "
    "Réponds en JSON : "
    '{"reasoning_chain": ["...", "..."], "answer": "réponse courte"}'
)

# 学习曲线的数据点数量
CURVE_POINTS = [10, 50, 200, 500, 1000, 3000]


def extract_supporting_sentences(example: dict) -> list[tuple[str, str]]:
    """提取 HotpotQA 中的标准支持句。

    按出现顺序返回（标题、句子文本）列表。
    """
    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles = example.get("supporting_facts", {}).get("title", [])
    sf_sent_ids = example.get("supporting_facts", {}).get("sent_id", [])

    # 建立标题到句子列表的索引
    title_to_sents = {}
    for t, s in zip(titles, sentences):
        title_to_sents[t] = s

    # 提取支持句
    support_pairs = []
    seen = set()
    for sf_title, sf_idx in zip(sf_titles, sf_sent_ids):
        key = (sf_title, sf_idx)
        if key in seen:
            continue
        seen.add(key)
        sents = title_to_sents.get(sf_title, [])
        if 0 <= sf_idx < len(sents):
            support_pairs.append((sf_title, sents[sf_idx]))

    return support_pairs


def build_reasoning_chain(
    support_pairs: list[tuple[str, str]],
    question: str,
    answer: str,
    q_type: str,
) -> list[str]:
    """根据标准支持事实构建推理链。

    根据事实数量和问题类型，生成两到四个步骤。
    """
    if not support_pairs:
        return [f"La réponse à la question est {answer}."]

    chain = []

    # 事实步骤：每条支持事实对应一步
    for i, (title, sent) in enumerate(support_pairs):
        sent_clean = sent.strip().rstrip(".")
        chain.append(f"D'après l'article « {title} » : {sent_clean}.")

    # 归纳步骤
    if q_type == "comparison" and len(support_pairs) >= 2:
        t1 = support_pairs[0][0]
        t2 = support_pairs[1][0]
        chain.append(
            f"En comparant les informations sur {t1} et {t2}, "
            f"la réponse est {answer}."
        )
    elif len(support_pairs) >= 2:
        chain.append(
            f"En reliant ces {len(support_pairs)} faits, "
            f"la réponse est {answer}."
        )
    else:
        chain.append(f"Donc la réponse est {answer}.")

    return chain


def format_multihop_sample_v2(example: dict) -> dict | None:
    """将 HotpotQA 样本格式化为包含推理链的对话样本。

    V2 使用标准支持事实替换占位文本。
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    # 保留完整上下文（与 V1 一样，包含全部段落）
    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    context_parts = []
    for title, sents in zip(titles, sentences):
        context_parts.append(f"{title}: {' '.join(sents)}")
    context = "\n".join(context_parts)  # 全部段落，包括标准事实和干扰内容

    if not context:
        return None

    # 根据标准支持事实构建推理链
    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain(support_pairs, question, answer, q_type)

    user_prompt = f"Faits :\n{context}\n\nQuestion : {question}"
    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """用标准支持事实的推理链重建 V2 多跳数据。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    # 格式化全部样本
    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v2(ex)
        if sample:
            # 统计推理链长度
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V2 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    # 保存主数据文件。
    # 命名为 multihop_v2_train.jsonl，以匹配 learning_curve.py
    # 查找的 {task}_train.jsonl 模式，其中 task="multihop_v2"。
    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v2_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    # 为学习曲线创建不同大小的子样本集
    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v2_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # 统计该子样本集中的推理链
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # 展示一个样本
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V2 ===")
    logger.info("Question : %s", ex["messages"][1]["content"][-200:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])

    # 与 V1 比较
    v1_path = os.path.join(QLORA_DATA_DIR, "multihop_train.jsonl")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1_first = json.loads(f.readline())
        v1_parsed = json.loads(v1_first["messages"][2]["content"])
        logger.info("\n=== Exemple V1 (dégénéré) ===")
        logger.info("Chaîne   : %s", v1_parsed["reasoning_chain"])
        logger.info("Réponse  : %s", v1_parsed["answer"])


# ═══════════════════════════════════════════════════════════════════════
# V3：精简推理链与上下文（2 条标准事实、3 条干扰信息）
# ═══════════════════════════════════════════════════════════════════════

def build_reasoning_chain_v3(
    support_pairs: list[tuple[str, str]],
    answer: str,
    q_type: str,
) -> list[str]:
    """生成简短推理链：归纳关键事实，不逐字引用。"""
    if not support_pairs:
        return [f"Réponse : {answer}."]

    chain = []
    for title, sent in support_pairs:
        # 将过长句子截断到约 100 个字符
        s = sent.strip().rstrip(".")
        if len(s) > 120:
            s = s[:117] + "..."
        chain.append(f"{title} : {s}.")

    # 归纳
    if q_type == "comparison" and len(support_pairs) >= 2:
        chain.append(f"Comparaison → {answer}.")
    elif len(support_pairs) >= 2:
        chain.append(f"Donc → {answer}.")
    else:
        chain.append(f"Donc → {answer}.")

    return chain


def format_multihop_sample_v3(example: dict) -> dict | None:
    """V3：精简上下文（2 条标准事实、3 条干扰信息）和推理链。

    目标是每个样本总长度少于 1024 个令牌。
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles_set = set(example.get("supporting_facts", {}).get("title", []))

    # 区分标准事实段落与干扰段落
    gold_parts = []
    distractor_parts = []
    for title, sents in zip(titles, sentences):
        para = f"{title}: {' '.join(sents)}"
        if title in sf_titles_set:
            gold_parts.append(para)
        else:
            distractor_parts.append(para)

    # 保留两个标准事实段落和最多三个干扰段落，并打乱顺序
    selected = gold_parts + distractor_parts[:3]
    random.shuffle(selected)
    context = "\n".join(selected)

    if not context:
        return None

    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain_v3(support_pairs, answer, q_type)

    user_prompt = f"Faits :\n{context}\n\nQuestion : {question}"
    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data_v3(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """重建简短且符合令牌预算的 V3 多跳数据。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    random.seed(seed)

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v3(ex)
        if sample:
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V3 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v3_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v3_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # 示例
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V3 ===")
    logger.info("User (last 300): %s", ex["messages"][1]["content"][-300:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])


# ═══════════════════════════════════════════════════════════════════════
# V4：训练格式与 exp_multihop.py 的评估格式一致
#   修正 1：使用相同的系统提示词（含两个少样本示例）
#   修正 2：用户提示词按问题、支持事实列表、指令的顺序组织
#   修正 3：与 V3 一样缩减上下文（2 条标准事实、3 条干扰信息）
# ═══════════════════════════════════════════════════════════════════════

# 系统提示词与 exp_multihop.py 中的 SYSTEM_PROMPT 一致
MULTIHOP_SYSTEM_V4 = (
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
)


def format_multihop_sample_v4(example: dict) -> dict | None:
    """V4：与 exp_multihop.py 的系统提示词及用户提示词格式一致。

    - 系统提示词：使用 exp_multihop.py 的 SYSTEM_PROMPT（含少样本示例）。
    - 用户提示词：问题、支持事实列表、指令。
    - 精简上下文：2 条标准事实、3 条干扰信息（同 V3）。
    - 精简推理链（同 V3）。
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles_set = set(example.get("supporting_facts", {}).get("title", []))

    # 区分标准事实段落与干扰段落
    gold_parts = []
    distractor_parts = []
    for title, sents in zip(titles, sentences):
        # 每个句子作为独立事实，以列表格式呈现，与评估时一致
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if title in sf_titles_set:
                gold_parts.append(s)
            else:
                distractor_parts.append(s)

    # 保留全部标准事实句和最多十个干扰句
    selected = gold_parts + distractor_parts[:10]
    random.shuffle(selected)

    if not selected:
        return None

    # 使用与 exp_multihop.py 相同的项目列表格式
    facts = "\n".join(f"- {s}" for s in selected)

    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain_v3(support_pairs, answer, q_type)

    # 按 exp_multihop.py 的格式组织：问题、事实、指令
    user_prompt = (
        f"Question : {question}\n\n"
        f"Faits de support :\n{facts}\n\n"
        "Raisonne étape par étape puis donne la réponse."
    )

    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM_V4},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data_v4(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """重建与评估输入格式一致的 V4 多跳数据。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    random.seed(seed)

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v4(ex)
        if sample:
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V4 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v4_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v4_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # 示例
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V4 ===")
    logger.info("System (100 chars): %s...", ex["messages"][0]["content"][:100])
    logger.info("User (last 400): %s", ex["messages"][1]["content"][-400:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruction données multihop")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--version", choices=["v2", "v3", "v4"], default="v4",
        help="Version des données à générer (défaut: v4)",
    )
    args = parser.parse_args()
    if args.version == "v2":
        rebuild_multihop_data(max_samples=args.max_samples, seed=args.seed)
    elif args.version == "v3":
        rebuild_multihop_data_v3(max_samples=args.max_samples, seed=args.seed)
    else:
        rebuild_multihop_data_v4(max_samples=args.max_samples, seed=args.seed)
