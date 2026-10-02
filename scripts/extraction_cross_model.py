#!/usr/bin/env python3
"""
使用 B1 基准测试中的最佳模型运行关系抽取 V3 实验。

复用 exp_extraction.py 的 V3 提示词、系统提示词、同义词和匹配逻辑，
对每个模型直接调用 Ollama。每个模型使用独立检查点，避免冲突。

用法：
    # 测试 B1 排名前三的模型
    python scripts/extraction_cross_model.py

    # 测试单个模型
    python scripts/extraction_cross_model.py --models mistral-small:latest

    # 使用较小样本集快速测试
    python scripts/extraction_cross_model.py --n 50
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

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(WORKSPACE, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from synsynth_data import load_extraction_data
from synsynth_stats import bootstrap_ci

RESULTS_DIR = os.path.join(WORKSPACE, "results")
RESULTS_CROSS_DIR = os.path.join(RESULTS_DIR, "extraction_cross_model")
OLLAMA_BASE = "http://127.0.0.1:11434"
TIMEOUT = 600

# B1 中表现最好的关系抽取模型（原始结果，未采用 V3）
TEST_MODELS = [
    "mistral-small:latest",   # 原始 F1=0.666，B1 中最佳
    "gpt-oss:20b",            # 原始 F1=0.659
    "phi4-reasoning:plus",    # 原始 F1=0.656
]

# ── Wikidata 标签与同义词，与 exp_extraction.py V3 一致 ──────────────
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

# V3 系统提示词，与 exp_extraction.py 一致
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

# V3 同义词，与 exp_extraction.py 一致
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


# ── 解析与匹配，与 exp_extraction.py V3 一致 ──────────────────────────

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
    """V3 宽松匹配：Wikidata 标签、子串及同义词。"""
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


# ── 逐模型评估，使用独立检查点 ────────────────────────────────────────

def evaluate_model(model: str, data: list[dict]) -> dict:
    """使用该模型运行抽取 V3 评估，并单独保存检查点。"""
    safe = model.replace(":", "_").replace("/", "_")
    os.makedirs(RESULTS_CROSS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_CROSS_DIR, f"extraction_{safe}.json")

    details = []
    tp = fp = fn = 0
    start_idx = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        details = ckpt.get("details", [])
        tp = ckpt.get("true_positives", 0)
        fp = ckpt.get("false_positives", 0)
        fn = ckpt.get("false_negatives", 0)
        start_idx = len(details)
        if start_idx >= len(data):
            print(f"  [SKIP] {model} — déjà terminé ({start_idx}/{len(data)})")
            return ckpt

    print(f"\n{'='*60}")
    print(f"Extraction V3 cross-model: {model}")
    print(f"  Reprise à {start_idx}/{len(data)}")
    print(f"{'='*60}")

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        gold_rel = sample["relation"]
        gold_label = _WIKIDATA_LABELS.get(gold_rel, gold_rel)

        # V3 提示词，与 exp_extraction.py 一致
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

        # 失败时重试一次
        if pred is None:
            messages_retry = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt +
                 "\nRéponds UNIQUEMENT avec le JSON, rien d'autre."},
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
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            print(f"  {model}: {n_done}/{len(data)} "
                  f"P={p:.3f} R={r:.3f} F1={f1:.3f} ({elapsed:.0f}s)")

            result = _compile(details, model, tp, fp, fn, elapsed)
            with open(ckpt_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    return _compile(details, model, tp, fp, fn, elapsed)


def _compile(details, model, tp, fp, fn, elapsed):
    n = len(details)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    per_sample = [1.0 if d.get("status") == "correct" else 0.0 for d in details]
    return {
        "model": model,
        "experiment": "extraction_v3_cross_model",
        "n_samples": n,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy_ci": bootstrap_ci(per_sample) if n >= 10 else None,
        "elapsed_seconds": round(elapsed, 1),
        "details": details,
    }


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test V3 extraction avec différents modèles")
    parser.add_argument("--models", nargs="+", default=TEST_MODELS,
                        help="Modèles à tester")
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre d'échantillons (default: 500)")
    args = parser.parse_args()

    os.makedirs(RESULTS_CROSS_DIR, exist_ok=True)
    data = load_extraction_data(args.n) if args.n else load_extraction_data()
    print(f"Échantillons extraction : {len(data)}")

    all_results = []
    for model in args.models:
        result = evaluate_model(model, data)
        all_results.append(result)

    # ── 对比表 ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARAISON — Extraction V3 cross-model")
    print(f"{'='*80}\n")

    # 加载 gemma4 V3 基线结果
    baseline_path = os.path.join(RESULTS_DIR, "extraction.json")
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    # 加载 B1 原始结果
    b1_results = {}
    b1_dir = os.path.join(RESULTS_DIR, "extraction_benchmark")
    if os.path.exists(b1_dir):
        for f_name in os.listdir(b1_dir):
            if f_name.startswith("extraction_") and f_name.endswith(".json"):
                with open(os.path.join(b1_dir, f_name)) as f:
                    b1 = json.load(f)
                b1_results[b1.get("model", "")] = b1.get("f1", 0)

    header = (f"  {'Modèle':<25} {'P':>7} {'R':>7} {'F1 V3':>7} "
              f"{'F1 brut':>7} {'Δ V3':>7} {'Temps':>8}")
    print(header)
    print(f"  {'-'*72}")

    if baseline:
        b_p = baseline.get("precision", 0)
        b_r = baseline.get("recall", 0)
        b_f1 = baseline.get("f1_score", 0)
        b1_gemma = b1_results.get("gemma4:26b", 0)
        delta_g = f"+{b_f1 - b1_gemma:.3f}" if b1_gemma else "—"
        print(f"  {'gemma4:26b (pipeline)':<25} "
              f"{b_p:>7.3f} {b_r:>7.3f} {b_f1:>7.3f} "
              f"{b1_gemma:>7.3f} {delta_g:>7} "
              f"{baseline.get('elapsed_seconds', 0):>7.0f}s")

    for r in all_results:
        model = r.get("model", "?")
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1 = r.get("f1_score", 0)
        b1_f1 = b1_results.get(model, 0)
        delta = f"+{f1 - b1_f1:.3f}" if b1_f1 else "—"
        elapsed = r.get("elapsed_seconds", 0)
        print(f"  {model:<25} {p:>7.3f} {rec:>7.3f} {f1:>7.3f} "
              f"{b1_f1:>7.3f} {delta:>7} {elapsed:>7.0f}s")

    # 保存汇总结果
    summary = []
    if baseline:
        summary.append({
            "model": "gemma4:26b",
            "precision": baseline.get("precision", 0),
            "recall": baseline.get("recall", 0),
            "f1_v3": baseline.get("f1_score", 0),
            "f1_brut_b1": b1_results.get("gemma4:26b", 0),
            "n_samples": baseline.get("n_samples", 0),
        })
    for r in all_results:
        model = r.get("model", "?")
        summary.append({
            "model": model,
            "precision": r.get("precision", 0),
            "recall": r.get("recall", 0),
            "f1_v3": r.get("f1_score", 0),
            "f1_brut_b1": b1_results.get(model, 0),
            "n_samples": r.get("n_samples", 0),
        })

    summary_path = os.path.join(RESULTS_CROSS_DIR,
                                "extraction_cross_model_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé → {summary_path}")


if __name__ == "__main__":
    main()
