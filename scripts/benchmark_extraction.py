#!/usr/bin/env python3
"""
B1：关系抽取（DocRED）的多模型基准测试。

在同一份 Re-DocRED 数据（500 个样本）上比较专用模型 gemma4:26b
与 phi4、qwen3 等通用模型。

用法：
    python scripts/benchmark_extraction.py                         # 所有模型
    python scripts/benchmark_extraction.py --models phi4:latest    # 单个模型
    python scripts/benchmark_extraction.py --n 100                 # 部分样本
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
RESULTS_DIR = os.path.join(WORKSPACE, "results", "extraction_benchmark")
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
from synsynth_stats import token_f1, bootstrap_ci
from synsynth_data import load_extraction_data

# ── Ollama 配置 ─────────────────────────────────────────────────────────
OLLAMA_BASE = "http://127.0.0.1:11434"
TIMEOUT = 600

# ── Wikidata 属性到标签的映射，与 exp_extraction.py 一致 ──────────────
_WIKIDATA_LABELS: dict[str, str] = {
    "P6": "head_of_government", "P17": "country", "P19": "place_of_birth",
    "P20": "place_of_death", "P22": "father", "P25": "mother",
    "P26": "spouse", "P27": "country_of_citizenship", "P30": "continent",
    "P31": "instance_of", "P35": "head_of_state", "P36": "capital",
    "P37": "official_language", "P39": "position_held", "P40": "child",
    "P47": "shares_border_with", "P50": "author", "P54": "member_of_sports_team",
    "P57": "director", "P58": "screenwriter", "P69": "educated_at",
    "P86": "composer", "P102": "member_of_political_party", "P108": "employer",
    "P112": "founded_by", "P118": "league", "P123": "publisher",
    "P127": "owned_by", "P131": "located_in_admin", "P136": "genre",
    "P137": "operator", "P140": "religion", "P150": "contains_admin",
    "P155": "follows", "P156": "followed_by", "P159": "headquarters_location",
    "P161": "cast_member", "P162": "producer", "P166": "award_received",
    "P170": "creator", "P171": "parent_taxon", "P172": "ethnic_group",
    "P175": "performer", "P176": "manufacturer", "P178": "developer",
    "P179": "series", "P190": "twinned_city", "P194": "legislative_body",
    "P205": "basin_country", "P206": "located_near_water",
    "P241": "military_branch", "P264": "record_label",
    "P272": "production_company", "P276": "location",
    "P279": "subclass_of", "P355": "subsidiary",
    "P361": "part_of", "P364": "original_language",
    "P400": "platform", "P403": "mouth_of_watercourse",
    "P449": "original_network", "P463": "member_of",
    "P488": "chairperson", "P495": "country_of_origin",
    "P527": "has_part", "P530": "diplomatic_relation",
    "P551": "residence", "P569": "date_of_birth",
    "P570": "date_of_death", "P571": "inception",
    "P576": "dissolved", "P577": "publication_date",
    "P580": "start_time", "P582": "end_time",
    "P607": "conflict", "P674": "characters",
    "P676": "lyrics_by", "P706": "located_on_terrain",
    "P710": "participant", "P737": "influenced_by",
    "P740": "location_of_formation", "P749": "parent_organization",
    "P800": "notable_work", "P807": "separated_from",
    "P840": "narrative_location", "P937": "work_location",
    "P1001": "jurisdiction", "P1056": "product",
    "P1198": "unemployment_rate", "P1336": "territory_claimed_by",
    "P1344": "participant_in", "P1365": "replaces",
    "P1366": "replaced_by", "P1376": "capital_of",
    "P1412": "languages_spoken", "P3373": "sibling",
}

_VALID_RELATIONS = sorted(set(_WIKIDATA_LABELS.values()))
_RELATIONS_LIST = ", ".join(_VALID_RELATIONS)

# ── 模型 ────────────────────────────────────────────────────────────────
MODELS = [
    "gemma4:26b",            # 关系抽取专用模型，基线 F1=0.702
    "phi4:latest",           # 14B 稠密模型
    "phi4-reasoning:plus",   # 14B 推理模型
    "qwen3:14b",             # 14B 支持推理模式
    "gpt-oss:20b",           # 20B 混合专家模型
    "mistral-small:latest",  # 22B 稠密模型
    "qwen3.5:27b",           # 27B 混合架构模型
    "deepseek-r1:32b",       # 32B 推理模型
]

# ── 系统提示词，与 exp_extraction.py 一致 ──────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in Relation Extraction. "
    "Given a text and two named entities (head and tail), "
    "identify the relation from head to tail. "
    "You MUST choose exactly one relation from this list:\n"
    f"{_RELATIONS_LIST}\n\n"
    "IMPORTANT RULES:\n"
    "- NEVER output 'no_relation', 'none', or 'unknown'. Always pick the closest relation.\n"
    "- For geographic/administrative entities (cities, regions, countries), "
    "prefer 'country', 'located_in_admin', 'contains_admin', or 'part_of'.\n"
    "- 'notable_work' means a creative work (book, film, song) by an author/artist. "
    "Do NOT use 'notable_work' for geographic, political, or family relations.\n"
    "- For family relations, use the specific relation: "
    "'father', 'mother', 'child', 'sibling', 'spouse'.\n"
    "Respond ONLY with JSON: "
    "{\"relation\": \"…\", \"confidence\": 0.0-1.0}"
)

# --- 同义词，与 exp_extraction.py 一致 ---
_SYNONYMS = {
    "start_time": {"year", "date", "start", "start_date", "began", "beginning", "from"},
    "end_time": {"year", "date", "end", "end_date", "ended", "until", "to"},
    "location": {"place", "located", "located_in", "situated", "venue", "city",
                 "held_in", "part_of", "located_in_admin"},
    "country": {"nation", "state", "located_in_country", "part_of", "located_in",
                "located_in_admin", "contains_admin", "is_part_of"},
    "located_in_admin": {"part_of", "located_in", "country", "contains_admin",
                         "is_part_of", "location"},
    "contains_admin": {"part_of", "has_part", "country", "located_in_admin",
                       "capital_of", "located_in"},
    "country_of_origin": {"country", "nationality", "located_in",
                          "original_language", "part_of"},
    "country_of_citizenship": {"nationality", "citizen", "citizen_of"},
    "place_of_birth": {"born", "born_in", "birthplace", "birth_place"},
    "place_of_death": {"died", "died_in", "death_place"},
    "date_of_birth": {"born", "birth", "birth_date", "born_on", "year"},
    "date_of_death": {"died", "death", "death_date", "died_on", "year"},
    "participant_in": {"participated", "competed", "took_part", "participant",
                       "competition", "conflict"},
    "participant": {"participant_in", "competed", "took_part"},
    "conflict": {"participant_in", "part_of"},
    "member_of": {"belongs_to", "part_of", "affiliated"},
    "instance_of": {"type", "is_a", "kind_of", "category"},
    "capital": {"capital_of", "capital_city"},
    "spouse": {"married", "husband", "wife", "married_to"},
    "educated_at": {"studied", "studied_at", "university", "school", "alma_mater"},
    "employer": {"works_for", "employed_by", "works_at"},
    "founded_by": {"founder", "created_by", "established_by"},
    "award_received": {"won", "received", "awarded", "prize"},
    "inception": {"founded_by", "founded_in", "established", "created",
                  "start_time", "establishment_year"},
    "jurisdiction": {"is_part_of", "part_of", "residence"},
}


# ── 调用 Ollama ─────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list[dict],
                json_format: bool = False) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 256,
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


# ── 解析与匹配 ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _parse_response(raw: str) -> dict | None:
    if not raw.strip():
        return None
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    m = re.search(r"\{[^}]*\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if "relation" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _relation_match(pred_rel: str, gold_rel: str) -> bool:
    pred_n = _normalize(pred_rel).replace(" ", "_")
    gold_n = _normalize(gold_rel).replace(" ", "_")

    if pred_n == gold_n:
        return True

    gold_label = _WIKIDATA_LABELS.get(gold_rel.upper(), "").lower()
    if not gold_label:
        gold_label = _WIKIDATA_LABELS.get(gold_rel, "").lower()

    if gold_label:
        if pred_n == gold_label.replace(" ", "_"):
            return True
        if pred_n in gold_label or gold_label in pred_n:
            return True
        pred_words = {w for w in pred_n.split("_") if len(w) >= 4}
        gold_words = {w for w in gold_label.split("_") if len(w) >= 4}
        if pred_words and gold_words and (pred_words & gold_words):
            return True
        gold_syns = _SYNONYMS.get(gold_label, set())
        if pred_n in gold_syns:
            return True

    return False


# ── 评估单个模型 ────────────────────────────────────────────────────────

def evaluate_model(model: str, data: list[dict],
                   checkpoint_path: str) -> dict:
    details = []
    tp = fp = fn = 0
    start_idx = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        tp = ckpt.get("tp", 0)
        fp = ckpt.get("fp", 0)
        fn = ckpt.get("fn", 0)
        start_idx = len(details)
        if start_idx >= len(data):
            print(f"  [SKIP] {model} — déjà terminé ({start_idx}/{len(data)})")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Extraction benchmark: {model}")
    print(f"{'='*60}")

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        gold_rel = sample["relation"]
        gold_label = _WIKIDATA_LABELS.get(gold_rel, gold_rel)

        prompt = (
            f"Texte : « {sample['text']} »\n\n"
            f"Entité head : {sample['head']}\n"
            f"Entité tail : {sample['tail']}\n\n"
            f"Quelle relation lie head à tail dans ce texte ? "
            f"Réponds en JSON : {{\"relation\": \"…\", \"confidence\": 0.0-1.0}}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = ollama_chat(model, messages, json_format=True)
        pred = _parse_response(raw)

        # 失败时重试
        if pred is None:
            messages_retry = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + "\nRéponds UNIQUEMENT avec le JSON, rien d'autre."},
            ]
            raw = ollama_chat(model, messages_retry, json_format=True)
            pred = _parse_response(raw)

        if pred is None:
            fn += 1
            details.append({"idx": i, "status": "parse_fail",
                            "raw": raw[:200], "gold_rel": gold_rel,
                            "gold_label": gold_label})
        else:
            rel_ok = _relation_match(pred.get("relation", ""), gold_rel)
            conf = pred.get("confidence", None)
            if isinstance(conf, (int, float)):
                conf = round(float(conf), 4)
            else:
                conf = None

            if rel_ok:
                tp += 1
                details.append({"idx": i, "status": "correct",
                                "pred_rel": pred.get("relation", ""),
                                "gold_rel": gold_rel, "gold_label": gold_label,
                                "confidence": conf})
            else:
                fp += 1
                fn += 1
                details.append({"idx": i, "status": "mismatch",
                                "pred_rel": pred.get("relation", ""),
                                "gold_rel": gold_rel, "gold_label": gold_label,
                                "confidence": conf})

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            elapsed = time.time() - t0
            n_done = i + 1
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            print(f"  {model}: {n_done}/{len(data)} "
                  f"(P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}, {elapsed:.0f}s)")

            result = _compile(details, tp, fp, fn, model,
                              time.time() - t0)
            with open(checkpoint_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    return _compile(details, tp, fp, fn, model, time.time() - t0)


def _compile(details, tp, fp, fn, model, elapsed) -> dict:
    n = len(details)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    per_sample = [1.0 if d.get("status") == "correct" else 0.0
                  for d in details]
    ci = bootstrap_ci(per_sample) if per_sample else {}
    parse_fails = sum(1 for d in details if d.get("status") == "parse_fail")
    return {
        "model": model,
        "n": n,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy_ci": ci,
        "parse_fail_pct": round(parse_fails / n * 100, 1) if n else 0,
        "elapsed_seconds": round(elapsed, 1),
        "details": details,
    }


# ── 汇总 ───────────────────────────────────────────────────────────────

def print_leaderboard(results: list[dict]):
    print("\n" + "="*80)
    print("LEADERBOARD B1 — Extraction de relations (Re-DocRED)")
    print("="*80)
    results_sorted = sorted(results, key=lambda r: r["f1"], reverse=True)
    header = (f"{'Modèle':<25} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'Parse%':>7} {'Temps':>7}")
    print(header)
    print("-" * 80)
    for r in results_sorted:
        print(f"{r['model']:<25} {r['precision']:.3f} {r['recall']:.3f} "
              f"{r['f1']:.3f} {r['parse_fail_pct']:>5.1f}% "
              f"{r['elapsed_seconds']:>6.0f}s")

    print()
    best = results_sorted[0]
    worst = results_sorted[-1]
    print(f"Meilleur : {best['model']} (F1={best['f1']:.3f})")
    print(f"Dernier  : {worst['model']} (F1={worst['f1']:.3f})")
    print(f"Écart    : {best['f1'] - worst['f1']:.3f}")


def save_summary(results: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []
    for r in results:
        summary.append({
            "model": r["model"],
            "n": r["n"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "accuracy_ci": r.get("accuracy_ci", {}),
            "parse_fail_pct": r["parse_fail_pct"],
            "elapsed_seconds": r["elapsed_seconds"],
        })

    path = os.path.join(RESULTS_DIR, "extraction_benchmark_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary → {path}")


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B1: Benchmark multi-modèle extraction de relations")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Modèles à évaluer")
    parser.add_argument("--n", type=int, default=500,
                        help="Nombre d'échantillons (default: 500)")
    args = parser.parse_args()

    print(f"B1 Extraction Benchmark — {args.n} échantillons")
    print(f"Modèles : {args.models}")

    data = load_extraction_data(args.n)
    print(f"Échantillons chargés : {len(data)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for model in args.models:
        safe = model.replace(":", "_").replace("/", "_")
        ckpt_path = os.path.join(RESULTS_DIR, f"extraction_{safe}.json")
        result = evaluate_model(model, data, ckpt_path)
        results.append(result)

    print_leaderboard(results)
    save_summary(results)


if __name__ == "__main__":
    main()
