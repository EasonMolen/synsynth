#!/usr/bin/env python3
"""
建议 C：对混淆矩阵自动聚类。

根据关系抽取结果构建预测值与标准值的混淆矩阵，
执行谱聚类，并将自动聚类结果与手工定义的 25 个同义词组比较。

用法：
    python confusion_clustering.py [--results path/to/extraction.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import RESULTS_DIR, logger, safe_path

OUTPUT_DIR = safe_path("results", "confusion_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 手工定义的同义词组，用作参考 ─────────────────────────────────────
MANUAL_SYNONYM_GROUPS = {
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

# Wikidata 标签到文本的映射；从 exp_extraction.py 复制以便独立运行
_WIKIDATA_LABELS = {}


def _normalize(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s.strip().lower()).replace(" ", "_")


def load_confusion_data(results_paths: list[str]) -> list[dict]:
    """从一个或多个文件加载（预测关系、标准标签）数据。"""
    pairs = []
    for path in results_paths:
        with open(path) as f:
            data = json.load(f)
        for d in data.get("details", []):
            if d.get("status") in ("correct", "mismatch"):
                pred = _normalize(d.get("pred_rel", ""))
                gold = _normalize(d.get("gold_label", d.get("gold_rel", "")))
                if pred and gold:
                    pairs.append({"pred": pred, "gold": gold,
                                  "correct": d["status"] == "correct"})
    logger.info("Paires chargées : %d (depuis %d fichiers)", len(pairs), len(results_paths))
    return pairs


def build_confusion_matrix(pairs: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """构建归一化混淆矩阵 C[标准值, 预测值]。"""
    gold_counts = Counter(p["gold"] for p in pairs)
    pred_counts = Counter(p["pred"] for p in pairs)

    # 标签包括全部标准值以及出现至少三次的预测值
    all_labels = sorted(set(
        [g for g, c in gold_counts.items() if c >= 2] +
        [p for p, c in pred_counts.items() if c >= 3]
    ))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    C = np.zeros((n, n), dtype=np.float64)
    for p in pairs:
        gi = label_to_idx.get(p["gold"])
        pi = label_to_idx.get(p["pred"])
        if gi is not None and pi is not None:
            C[gi, pi] += 1

    # 按行归一化，得到 P(预测值|标准值)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C_norm = C / row_sums

    logger.info("Matrice de confusion : %d × %d labels", n, n)
    return C_norm, all_labels, all_labels


def spectral_clustering_on_confusion(C_norm: np.ndarray, labels: list[str],
                                      n_clusters: int | None = None) -> dict:
    """对对称化后的混淆矩阵执行谱聚类。"""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    # 对称化：相似度 = C + C 的转置，表示双向混淆
    S = (C_norm + C_norm.T) / 2
    np.fill_diagonal(S, 0)  # 忽略标签自身的混淆值

    # 未指定聚类数时，搜索最佳 k 值
    if n_clusters is None:
        best_k, best_score = 2, -1
        for k in range(2, min(20, len(labels))):
            try:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                        random_state=42, n_init=10)
                cluster_labels = sc.fit_predict(S + 1e-10)
                if len(set(cluster_labels)) < 2:
                    continue
                score = silhouette_score(S, cluster_labels, metric="precomputed")
                if score > best_score:
                    best_k, best_score = k, score
            except Exception:
                continue
        n_clusters = best_k
        logger.info("Meilleur k=%d (silhouette=%.3f)", n_clusters, best_score)

    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                            random_state=42, n_init=10)
    cluster_labels = sc.fit_predict(S + 1e-10)

    # 构建聚类分组
    groups = defaultdict(list)
    for label, cl in zip(labels, cluster_labels):
        groups[int(cl)].append(label)

    return {
        "n_clusters": n_clusters,
        "silhouette": round(float(silhouette_score(S, cluster_labels, metric="precomputed")), 4)
                      if len(set(cluster_labels)) >= 2 else 0,
        "clusters": dict(groups),
    }


def hierarchical_clustering_on_confusion(C_norm: np.ndarray, labels: list[str],
                                          distance_threshold: float = 0.7) -> dict:
    """对混淆矩阵执行凝聚层次聚类。"""
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import squareform

    # 距离 = 1 - 相似度
    S = (C_norm + C_norm.T) / 2
    np.fill_diagonal(S, 1)
    D = 1 - S
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, None)

    hc = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    cluster_labels = hc.fit_predict(D)

    groups = defaultdict(list)
    for label, cl in zip(labels, cluster_labels):
        groups[int(cl)].append(label)

    # 过滤只含一个元素的分组
    non_singleton = {k: v for k, v in groups.items() if len(v) >= 2}

    return {
        "n_clusters": len(groups),
        "n_non_singleton": len(non_singleton),
        "distance_threshold": distance_threshold,
        "clusters": dict(groups),
        "non_singleton_clusters": non_singleton,
    }


def compare_with_manual(auto_clusters: dict[int, list[str]]) -> dict:
    """比较自动聚类结果与手工分组。"""
    # 根据 Jaccard 相似度为每个自动分组找出最接近的手工分组
    matches = []
    manual_groups_flat = []
    for gold_key, syns in MANUAL_SYNONYM_GROUPS.items():
        manual_groups_flat.append((gold_key, {gold_key} | syns))

    for cl_id, cl_members in auto_clusters.items():
        cl_set = set(cl_members)
        best_jaccard = 0
        best_match = None
        for gold_key, gold_set in manual_groups_flat:
            jaccard = len(cl_set & gold_set) / len(cl_set | gold_set) if (cl_set | gold_set) else 0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = gold_key

        matches.append({
            "cluster_id": cl_id,
            "cluster_members": sorted(cl_members),
            "best_manual_match": best_match,
            "jaccard": round(best_jaccard, 4),
            "overlap": sorted(cl_set & (MANUAL_SYNONYM_GROUPS.get(best_match, set()) | {best_match}))
                       if best_match else [],
        })

    avg_jaccard = np.mean([m["jaccard"] for m in matches]) if matches else 0
    return {
        "n_auto_clusters": len(auto_clusters),
        "n_manual_groups": len(MANUAL_SYNONYM_GROUPS),
        "avg_jaccard": round(float(avg_jaccard), 4),
        "matches": matches,
    }


def evaluate_with_auto_synonyms(pairs: list[dict], auto_clusters: dict[int, list[str]]) -> dict:
    """将自动聚类分组用作同义词组来评估 F1。

    比较完全匹配、手工同义词匹配和自动分组匹配。
    """
    # 建立查询表：记录每个标签在同一分组中的其他标签
    label_to_synonyms_auto = defaultdict(set)
    for members in auto_clusters.values():
        for m in members:
            label_to_synonyms_auto[m] = set(members) - {m}

    label_to_synonyms_manual = {}
    for gold_key, syns in MANUAL_SYNONYM_GROUPS.items():
        label_to_synonyms_manual[gold_key] = syns

    results = {"exact": {"tp": 0, "fp": 0, "fn": 0},
               "manual_synonyms": {"tp": 0, "fp": 0, "fn": 0},
               "auto_synonyms": {"tp": 0, "fp": 0, "fn": 0}}

    for p in pairs:
        pred, gold = p["pred"], p["gold"]

        # 完全匹配
        if pred == gold:
            results["exact"]["tp"] += 1
        else:
            results["exact"]["fp"] += 1
            results["exact"]["fn"] += 1

        # 手工同义词匹配
        manual_syns = label_to_synonyms_manual.get(gold, set())
        if pred == gold or pred in manual_syns:
            results["manual_synonyms"]["tp"] += 1
        else:
            results["manual_synonyms"]["fp"] += 1
            results["manual_synonyms"]["fn"] += 1

        # 自动分组匹配
        auto_syns = label_to_synonyms_auto.get(gold, set())
        if pred == gold or pred in auto_syns:
            results["auto_synonyms"]["tp"] += 1
        else:
            results["auto_synonyms"]["fp"] += 1
            results["auto_synonyms"]["fn"] += 1

    # 计算 F1
    for method in results:
        m = results[method]
        prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        m["precision"] = round(prec, 4)
        m["recall"] = round(rec, 4)
        m["f1"] = round(f1, 4)

    return results


def plot_confusion_heatmap(C_norm: np.ndarray, labels: list[str],
                            top_n: int = 30, label: str = "extraction"):
    """绘制混淆矩阵热力图，显示出现频率最高的 N 个标签。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 根据行列之和选择最活跃的 N 个标签
    activity = C_norm.sum(axis=0) + C_norm.sum(axis=1)
    top_idx = np.argsort(-activity)[:top_n]
    C_sub = C_norm[np.ix_(top_idx, top_idx)]
    sub_labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(C_sub, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(sub_labels)))
    ax.set_yticks(range(len(sub_labels)))
    ax.set_xticklabels(sub_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(sub_labels, fontsize=8)
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Gold", fontsize=12)
    ax.set_title(f"Matrice de confusion normalisée — {label}\n"
                 f"(top {top_n} relations)", fontsize=13)
    fig.colorbar(im, ax=ax, label="P(pred | gold)")

    fig_path = os.path.join(OUTPUT_DIR, f"confusion_heatmap_{label}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def plot_dendrogram(C_norm: np.ndarray, labels: list[str], label: str = "extraction"):
    """绘制层次聚类树状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    S = (C_norm + C_norm.T) / 2
    np.fill_diagonal(S, 1)
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0)

    # 只保留实际出现的标签
    activity = C_norm.sum(axis=0) + C_norm.sum(axis=1)
    active = activity > 0.01
    D_sub = D[np.ix_(active, active)]
    sub_labels = [l for l, a in zip(labels, active) if a]

    if len(sub_labels) < 3:
        logger.warning("Pas assez de labels actifs pour le dendrogramme.")
        return None

    condensed = squareform(D_sub, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, labels=sub_labels, ax=ax, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7)
    ax.set_ylabel("Distance (1 - confusion)", fontsize=12)
    ax.set_title(f"Dendrogramme des confusions — {label}", fontsize=13)
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Seuil = 0.7")
    ax.legend()

    fig_path = os.path.join(OUTPUT_DIR, f"dendrogram_{label}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def analyze(results_paths: list[str], label: str = "all") -> dict:
    """执行完整分析：混淆矩阵、聚类、比较和评估。"""
    pairs = load_confusion_data(results_paths)
    if len(pairs) < 20:
        logger.error("Pas assez de données de confusion (%d).", len(pairs))
        return {"error": "insufficient_data"}

    C_norm, row_labels, col_labels = build_confusion_matrix(pairs)

    # 谱聚类
    spectral = spectral_clustering_on_confusion(C_norm, row_labels)
    logger.info("Spectral clustering : %d clusters, silhouette=%.3f",
                spectral["n_clusters"], spectral["silhouette"])

    # 层次聚类
    hierarchical = hierarchical_clustering_on_confusion(C_norm, row_labels)
    logger.info("Hierarchical clustering : %d clusters (%d non-singleton)",
                hierarchical["n_clusters"], hierarchical["n_non_singleton"])

    # 与手工分组比较
    comparison_spectral = compare_with_manual(spectral["clusters"])
    comparison_hier = compare_with_manual(hierarchical.get("non_singleton_clusters", {}))

    # 使用自动聚类匹配结果评估 F1
    eval_spectral = evaluate_with_auto_synonyms(pairs, spectral["clusters"])
    eval_hier = evaluate_with_auto_synonyms(pairs, hierarchical.get("non_singleton_clusters", {}))

    # 绘图
    fig_heatmap = plot_confusion_heatmap(C_norm, row_labels, label=label)
    fig_dendro = plot_dendrogram(C_norm, row_labels, label=label)

    result = {
        "label": label,
        "n_pairs": len(pairs),
        "n_labels": len(row_labels),
        "spectral_clustering": spectral,
        "hierarchical_clustering": hierarchical,
        "comparison_spectral_vs_manual": comparison_spectral,
        "comparison_hierarchical_vs_manual": comparison_hier,
        "evaluation": {
            "exact_match": eval_spectral["exact"],
            "manual_synonyms": eval_spectral["manual_synonyms"],
            "auto_spectral": eval_spectral["auto_synonyms"],
            "auto_hierarchical": eval_hier["auto_synonyms"],
        },
        "figures": [fig_heatmap, fig_dendro],
    }

    out_path = os.path.join(OUTPUT_DIR, f"confusion_analysis_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Résultat → %s", out_path)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clustering de la matrice de confusion (Suggestion C)")
    parser.add_argument("--results", type=str, nargs="+", default=None,
                        help="Fichier(s) de résultats d'extraction.")
    parser.add_argument("--label", type=str, default="all")
    args = parser.parse_args()

    if args.results is None:
        # 加载所有可用的关系抽取结果
        args.results = []
        for fname in ["extraction.json", "extraction_qlora.json"]:
            p = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(p):
                args.results.append(p)

    if not args.results:
        logger.error("Aucun fichier de résultats trouvé.")
        sys.exit(1)

    analyze(args.results, args.label)


if __name__ == "__main__":
    main()
