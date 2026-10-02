#!/usr/bin/env python3
"""
使用多跳推理 V4 的难题，对八个大语言模型进行零样本基准测试。

用法：
    python scripts/benchmark_model_selection.py                          # 181 道难题
    python scripts/benchmark_model_selection.py --models phi4 qwen3:14b  # 指定模型
    python scripts/benchmark_model_selection.py --full                    # 全部 500 道题
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

# ── 路径 ────────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFICULT_PATH = os.path.join(WORKSPACE, "results", "model_selection", "difficult_questions.json")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "model_selection")
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
from synsynth_stats import token_f1

# ── Ollama 配置 ─────────────────────────────────────────────────────────
OLLAMA_BASE = "http://127.0.0.1:11434"
TIMEOUT = 600

# ── 待评估模型 ──────────────────────────────────────────────────────────
MODELS = [
    "phi4:latest",           # 14B 稠密模型，基线
    "phi4-reasoning:plus",   # 14B 稠密模型，经过 o3-mini 监督微调与强化学习
    "qwen3:14b",             # 14B 稠密模型，支持推理模式
    "gemma4:26b",            # 26B 混合专家模型，激活约 3.8B 参数
    "gpt-oss:20b",           # 20B 混合专家模型，OpenAI 开放权重
    "magistral:24b",         # 24B 稠密模型，Mistral 推理模型
    "qwen3.5:27b",           # 27B 混合架构模型，当时的较新模型
    "deepseek-r1:32b",       # 32B 稠密模型，使用蒸馏推理能力
]

# ── 系统提示词，与 V4 评估保持一致 ────────────────────────────────────
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
)


# ── 调用 Ollama ─────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list[dict], json_format: bool = False) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
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


# ── 解析回答 ────────────────────────────────────────────────────────────

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

    # 去掉推理模型输出的 <think>...</think> 部分
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw
    cleaned = _strip_markdown(cleaned)

    # 优先直接解析 JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            obj["answer"] = str(obj["answer"])
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # 根据花括号嵌套层级解析
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

    # 解析失败时用正则提取
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
    if m:
        answer = m.group(1).strip()
        chain_match = re.findall(r'"reasoning_chain"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        chain = []
        if chain_match:
            chain = re.findall(r'"([^"]+)"', chain_match[0])
        return {"answer": answer, "reasoning_chain": chain}

    # 最后按纯文本处理
    answer = _extract_short_answer(cleaned)
    return {"answer": answer, "reasoning_chain": []}


# ── 答案匹配 ────────────────────────────────────────────────────────────

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


# ── 加载数据 ────────────────────────────────────────────────────────────

def load_difficult_questions() -> list[dict]:
    with open(DIFFICULT_PATH) as f:
        return json.load(f)


def load_full_eval_data() -> list[dict]:
    v4_path = os.path.join(WORKSPACE, "results", "learning_curve",
                           "learning_curve_multihop_v4.json")
    with open(v4_path) as f:
        v4 = json.load(f)
    first = v4[0]
    return [{"idx": d["idx"], "question": d["question"],
             "gold_answer": d["gold_answer"]}
            for d in first["details"]]


def load_supporting_facts() -> dict[int, list[str]]:
    """从 HotpotQA 数据集中加载题号到支持事实的映射。"""
    sys.path.insert(0, SCRIPTS_DIR)
    from synsynth_data import load_multihop_data
    data = load_multihop_data(500)
    return {i: d.get("supporting_facts", []) for i, d in enumerate(data)}


# ── 评估单个模型 ────────────────────────────────────────────────────────

def evaluate_model(model: str, questions: list[dict],
                   facts_by_idx: dict[int, list[str]],
                   checkpoint_path: str) -> dict:
    details = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        start_idx = len(details)
        if start_idx >= len(questions):
            print(f"  [SKIP] {model} — déjà terminé ({start_idx}/{len(questions)})")
            return ckpt

    system = SYSTEM_PROMPT + (
        "\nTu dois répondre UNIQUEMENT avec un objet JSON valide, "
        "sans texte avant ni après."
    )

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
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        raw = ollama_chat(model, messages, json_format=True)
        pred = parse_response(raw)

        gold = q["gold_answer"]
        ans = pred["answer"] if pred else ""
        em = answer_match(ans, gold) if pred else False
        f1 = token_f1(ans, gold) if pred else 0.0
        chain = pred.get("reasoning_chain", []) if pred else []
        chain_len = len(chain) if isinstance(chain, list) else 0

        details.append({
            "idx": idx,
            "question": q["question"],
            "gold_answer": gold,
            "pred_answer": ans,
            "exact_match": em,
            "token_f1": round(f1, 4),
            "chain_length": chain_len,
            "raw_response_len": len(raw),
        })

        if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
            elapsed = time.time() - t0
            n_done = qi + 1
            em_so_far = sum(1 for d in details if d["exact_match"]) / len(details)
            f1_so_far = sum(d["token_f1"] for d in details) / len(details)
            print(f"  {model}: {n_done}/{len(questions)} "
                  f"(EM={em_so_far:.3f}, F1={f1_so_far:.3f}, "
                  f"{elapsed:.0f}s)")

            result = _compute_metrics(details, model)
            with open(checkpoint_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compute_metrics(details, model)


def _compute_metrics(details: list[dict], model: str) -> dict:
    n = len(details)
    if n == 0:
        return {"model": model, "n": 0, "em": 0, "f1": 0, "details": []}
    em = sum(1 for d in details if d["exact_match"]) / n
    f1 = sum(d["token_f1"] for d in details) / n
    chain_avg = sum(d["chain_length"] for d in details) / n
    json_brut = sum(1 for d in details if d["chain_length"] == 0) / n
    return {
        "model": model,
        "n": n,
        "em": round(em, 4),
        "f1": round(f1, 4),
        "chain_avg": round(chain_avg, 2),
        "json_brut_pct": round(json_brut * 100, 1),
        "details": details,
    }


# ── 主流程 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on difficult multihop questions")
    parser.add_argument("--models", nargs="*", help="Subset of models to test")
    parser.add_argument("--full", action="store_true",
                        help="Use all 500 eval questions instead of difficult subset")
    args = parser.parse_args()

    models = args.models if args.models else MODELS

    if args.full:
        questions = load_full_eval_data()
        suffix = "full"
        print(f"=== Benchmark sur {len(questions)} questions (dataset complet) ===")
    else:
        questions = load_difficult_questions()
        suffix = "difficult"
        print(f"=== Benchmark sur {len(questions)} questions difficiles ===")

    print("Chargement des faits de support...")
    facts_by_idx = load_supporting_facts()
    print(f"  {len(facts_by_idx)} questions avec faits chargées.")

    all_results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Modèle : {model}")
        print(f"{'='*60}")

        ckpt_path = os.path.join(
            RESULTS_DIR,
            f"benchmark_{model.replace(':', '_').replace('/', '_')}_{suffix}.json")

        result = evaluate_model(model, questions, facts_by_idx, ckpt_path)
        all_results.append({
            "model": result["model"],
            "n": result["n"],
            "em": result["em"],
            "f1": result["f1"],
            "chain_avg": result["chain_avg"],
            "json_brut_pct": result["json_brut_pct"],
        })

        print(f"\n  → {model}: EM={result['em']:.3f}, F1={result['f1']:.3f}, "
              f"chain={result['chain_avg']:.1f}, JSON_brut={result['json_brut_pct']:.0f}%")

    # 汇总结果
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"{'Modèle':<30} {'EM':>6} {'F1':>6} {'Chain':>6} {'JSON%':>6}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x["em"], reverse=True):
        print(f"{r['model']:<30} {r['em']:>6.3f} {r['f1']:>6.3f} "
              f"{r['chain_avg']:>6.1f} {r['json_brut_pct']:>5.1f}%")

    summary_path = os.path.join(RESULTS_DIR, f"benchmark_summary_{suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()
