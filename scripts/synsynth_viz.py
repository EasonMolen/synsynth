"""
实验结果可视化。

生成汇总图表并保存到 results/。
"""
from __future__ import annotations

import os
from typing import Any

from synsynth_config import RESULTS_DIR, safe_path, logger


def plot_summary(all_results: dict[str, Any]) -> list[str]:
    """创建汇总图表，返回生成文件的路径列表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: list[str] = []

    # ── 图 1：各实验的主要指标 ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SYNSYNTH+ — Résultats Expérimentaux", fontsize=16, weight="bold")

    # 1a. 关系抽取
    ext = all_results.get("extraction", {})
    ax = axes[0, 0]
    metrics = ["precision", "recall", "f1_score"]
    values = [ext.get(m, 0.0) for m in metrics]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(["Précision", "Rappel", "F1-Score"], values, color=colors)
    ax.axhline(y=ext.get("target_f1", 0.85), color="red", linestyle="--",
               label=f"Cible F1 = {ext.get('target_f1', 0.85)}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp. 1 : Extraction de Relations")
    ax.legend()
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2%}", ha="center", fontsize=10)

    # 1b. 文本到查询
    qtq = all_results.get("text_to_query", {})
    ax = axes[0, 1]
    vals = [qtq.get("accuracy", 0), qtq.get("cypher_syntax_valid_rate", 0)]
    bars = ax.bar(["Accuracy", "Cypher valide"], vals, color=["#4C72B0", "#55A868"])
    ax.axhline(y=qtq.get("target_accuracy", 0.9), color="red", linestyle="--",
               label=f"Cible = {qtq.get('target_accuracy', 0.9)}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp. 2 : Text-to-Query")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2%}", ha="center", fontsize=10)

    # 1c. 多跳推理
    mh = all_results.get("multihop_reasoning", {})
    ax = axes[1, 0]
    vals = [mh.get("exact_accuracy", 0), mh.get("partial_accuracy", 0)]
    bars = ax.bar(["Exact Match", "Partiel"], vals, color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp. 3 : Raisonnement Multi-hop")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2%}", ha="center", fontsize=10)

    # 1d. RAGAS
    rag = all_results.get("rag_faithfulness", {})
    ax = axes[1, 1]
    ragas_labels = ["Fidélité", "Pertinence", "Préc. Contexte"]
    ragas_vals = [
        rag.get("avg_faithfulness", 0),
        rag.get("avg_answer_relevance", 0),
        rag.get("avg_context_precision", 0),
    ]
    bars = ax.bar(ragas_labels, ragas_vals, color=["#C44E52", "#55A868", "#8172B3"])
    ax.axhline(y=0.95, color="red", linestyle="--", label="Cible fidélité ≈ 1.0")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp. 4 : Évaluation RAGAS")
    ax.legend()
    for bar, v in zip(bars, ragas_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = safe_path("results", "synsynth_results_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    logger.info("Figure résumé → %s", p)

    # ── 图 2：四个评估方向的雷达图 ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ["Extraction\n(F1)", "Query\n(Accuracy)",
                   "Multi-hop\n(Exact)", "Faithfulness\n(RAGAS)"]
    scores = [
        ext.get("f1_score", 0),
        qtq.get("accuracy", 0),
        mh.get("exact_accuracy", 0),
        rag.get("avg_faithfulness", 0),
    ]
    targets = [0.85, 0.90, 0.70, 1.00]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_loop = scores + scores[:1]
    targets_loop = targets + targets[:1]
    angles_loop = angles + angles[:1]

    ax2.fill(angles_loop, scores_loop, color="#4C72B0", alpha=0.25)
    ax2.plot(angles_loop, scores_loop, "o-", color="#4C72B0",
             linewidth=2, label="Scores obtenus")
    ax2.plot(angles_loop, targets_loop, "s--", color="#C44E52",
             linewidth=2, label="Cibles SYNSYNTH+")
    ax2.set_xticks(angles)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("SYNSYNTH+ — Profil de Performance", fontsize=14,
                   weight="bold", pad=20)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))

    p2 = safe_path("results", "synsynth_radar.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    paths.append(p2)
    logger.info("Radar chart → %s", p2)

    return paths
