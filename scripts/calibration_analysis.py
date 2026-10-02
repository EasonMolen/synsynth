#!/usr/bin/env python3
"""
建议 E：校准模型自行报告的置信度。

分析关系抽取结果中的置信度，用于：
1. 绘制可靠性图
2. 计算期望校准误差（ECE）
3. 执行等距回归校准
4. 确定最佳拒绝阈值

用法：
    python calibration_analysis.py [--results path/to/extraction_qlora.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import RESULTS_DIR, logger, safe_path

OUTPUT_DIR = safe_path("results", "calibration")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_confidence_data(results_path: str) -> tuple[np.ndarray, np.ndarray]:
    """从关系抽取结果文件中加载（置信度、是否正确）数据。"""
    with open(results_path) as f:
        data = json.load(f)

    confidences = []
    corrects = []
    n_missing = 0

    for d in data.get("details", []):
        if d.get("status") == "parse_fail":
            continue
        conf = d.get("confidence")
        if conf is None:
            n_missing += 1
            continue
        confidences.append(float(conf))
        corrects.append(1.0 if d["status"] == "correct" else 0.0)

    if n_missing > 0:
        logger.warning("%d échantillons sans score de confiance (ignorés).", n_missing)

    logger.info("Données de calibration : %d échantillons avec confiance (sur %d details).",
                len(confidences), len(data.get("details", [])))
    return np.array(confidences), np.array(corrects)


def compute_ece(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> dict:
    """计算期望校准误差和各分箱的数据。"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # 最后一个分箱包含上界
            mask = (confidences >= lo) & (confidences <= hi)

        count = mask.sum()
        if count == 0:
            bins.append({
                "range": f"[{lo:.1f}, {hi:.1f}]",
                "count": 0, "avg_confidence": 0, "avg_accuracy": 0, "gap": 0
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = corrects[mask].mean()
        bins.append({
            "range": f"[{lo:.1f}, {hi:.1f}]",
            "count": int(count),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(abs(avg_conf - avg_acc)), 4),
        })

    # 按各分箱样本数加权计算 ECE
    ece = sum(b["count"] * b["gap"] for b in bins) / len(confidences) if len(confidences) > 0 else 0
    return {"ece": round(ece, 4), "n_bins": n_bins, "bins": bins}


def compute_auc_confidence(confidences: np.ndarray, corrects: np.ndarray) -> float:
    """将置信度用于预测正确性，计算 AUC-ROC。"""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(corrects)) < 2:
        return float("nan")
    return round(float(roc_auc_score(corrects, confidences)), 4)


def find_optimal_threshold(confidences: np.ndarray, corrects: np.ndarray) -> dict:
    """找出使 F1 最大的置信度阈值；低于阈值的预测被拒绝。"""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best = {"threshold": 0, "f1": 0, "precision": 0, "recall": 0, "n_kept": len(corrects)}

    for tau in thresholds:
        mask = confidences >= tau
        n_kept = mask.sum()
        if n_kept == 0:
            continue

        tp = corrects[mask].sum()
        fp = n_kept - tp
        fn = corrects[~mask].sum()  # 被错误拒绝的正确预测

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if f1 > best["f1"]:
            best = {
                "threshold": round(float(tau), 2),
                "f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "n_kept": int(n_kept),
                "n_rejected": int(len(corrects) - n_kept),
                "rejection_rate": round(float(1 - n_kept / len(corrects)), 4),
            }

    return best


def isotonic_calibration(confidences: np.ndarray, corrects: np.ndarray) -> dict:
    """执行等距回归校准，训练集和测试集按 70/30 划分。"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    idx_train, idx_test = train_test_split(
        np.arange(len(confidences)), test_size=0.3, random_state=42,
    )
    conf_train, corr_train = confidences[idx_train], corrects[idx_train]
    conf_test, corr_test = confidences[idx_test], corrects[idx_test]

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(conf_train, corr_train)

    cal_test = iso.predict(conf_test)

    # 比较校准前后的 ECE
    ece_before = compute_ece(conf_test, corr_test)["ece"]
    ece_after = compute_ece(cal_test, corr_test)["ece"]

    return {
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_reduction": round(ece_before - ece_after, 4),
        "n_train": len(idx_train),
        "n_test": len(idx_test),
    }


def plot_reliability_diagram(confidences: np.ndarray, corrects: np.ndarray,
                             ece_data: dict, label: str = "extraction"):
    """绘制可靠性图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = ece_data["bins"]
    non_empty = [b for b in bins if b["count"] > 0]

    confs = [b["avg_confidence"] for b in non_empty]
    accs = [b["avg_accuracy"] for b in non_empty]
    counts = [b["count"] for b in non_empty]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # 可靠性图
    ax1.bar(confs, accs, width=0.08, alpha=0.6, color="#2196F3", label="Accuracy réelle")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Calibration parfaite")
    ax1.set_ylabel("Accuracy réelle", fontsize=12)
    ax1.set_title(f"Diagramme de fiabilité — {label}\n"
                  f"ECE = {ece_data['ece']:.3f}", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # 置信度直方图
    ax2.bar(confs, counts, width=0.08, alpha=0.6, color="#FF9800")
    ax2.set_xlabel("Confiance prédite", fontsize=12)
    ax2.set_ylabel("Nombre", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig_path = os.path.join(OUTPUT_DIR, f"reliability_diagram_{label}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def plot_rejection_curve(confidences: np.ndarray, corrects: np.ndarray, label: str = "extraction"):
    """绘制 F1 和精确率随拒绝阈值变化的曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = np.arange(0.0, 1.01, 0.02)
    f1s, precs, recs, kept_pcts = [], [], [], []

    baseline_f1 = corrects.mean()  # 估算不拒绝预测时的 F1

    for tau in thresholds:
        mask = confidences >= tau
        n_kept = mask.sum()
        if n_kept == 0:
            f1s.append(0)
            precs.append(0)
            recs.append(0)
            kept_pcts.append(0)
            continue

        tp = corrects[mask].sum()
        fp = n_kept - tp
        fn = corrects[~mask].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        kept_pcts.append(n_kept / len(corrects) * 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, f1s, "-", color="#2196F3", linewidth=2, label="F1")
    ax1.plot(thresholds, precs, "--", color="#4CAF50", linewidth=1.5, label="Precision")
    ax1.plot(thresholds, recs, "--", color="#FF5722", linewidth=1.5, label="Recall")
    ax1.axhline(y=baseline_f1, color="gray", linestyle=":", label=f"Baseline (no reject) = {baseline_f1:.2f}")

    ax1.set_xlabel("Seuil de confiance τ", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title(f"Trade-off rejet/performance — {label}", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(thresholds, kept_pcts, alpha=0.1, color="orange")
    ax2.set_ylabel("% échantillons conservés", fontsize=10, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig_path = os.path.join(OUTPUT_DIR, f"rejection_curve_{label}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def analyze(results_path: str, label: str = "extraction") -> dict:
    """执行完整的置信度校准分析。"""
    confidences, corrects = load_confidence_data(results_path)

    if len(confidences) < 10:
        logger.error("Pas assez de données de confiance (%d). "
                      "Relancer l'extraction pour collecter les scores.", len(confidences))
        return {"error": "insufficient_confidence_data", "n_samples": len(confidences)}

    logger.info("Distribution des confiances : min=%.2f, max=%.2f, mean=%.2f, std=%.2f",
                confidences.min(), confidences.max(), confidences.mean(), confidences.std())

    # 1. ECE
    ece_data = compute_ece(confidences, corrects)
    logger.info("ECE = %.4f", ece_data["ece"])

    # 2. AUC-ROC
    auc = compute_auc_confidence(confidences, corrects)
    logger.info("AUC-ROC (confiance→correctness) = %.4f", auc)

    # 3. 最佳阈值
    opt = find_optimal_threshold(confidences, corrects)
    logger.info("Seuil optimal τ=%.2f → F1=%.4f (rejet %.1f%%)",
                opt["threshold"], opt["f1"], opt["rejection_rate"] * 100)

    # 4. 等距回归校准
    iso = isotonic_calibration(confidences, corrects)
    logger.info("Calibration isotonique : ECE %.4f → %.4f (réduction %.4f)",
                iso["ece_before"], iso["ece_after"], iso["ece_reduction"])

    # 5. 绘图
    fig_reliability = plot_reliability_diagram(confidences, corrects, ece_data, label)
    fig_rejection = plot_rejection_curve(confidences, corrects, label)

    # 完整结果
    result = {
        "label": label,
        "n_samples": len(confidences),
        "confidence_stats": {
            "min": round(float(confidences.min()), 4),
            "max": round(float(confidences.max()), 4),
            "mean": round(float(confidences.mean()), 4),
            "std": round(float(confidences.std()), 4),
            "median": round(float(np.median(confidences)), 4),
        },
        "ece": ece_data,
        "auc_roc": auc,
        "optimal_threshold": opt,
        "isotonic_calibration": iso,
        "figures": [fig_reliability, fig_rejection],
    }

    # 保存
    out_path = os.path.join(OUTPUT_DIR, f"calibration_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Résultat → %s", out_path)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyse de calibration de la confiance (Suggestion E)")
    parser.add_argument("--results", type=str, default=None,
                        help="Chemin du fichier de résultats d'extraction.")
    parser.add_argument("--label", type=str, default="extraction_qlora")
    args = parser.parse_args()

    if args.results is None:
        # 查找最新的结果
        candidates = [
            os.path.join(RESULTS_DIR, "extraction_qlora.json"),
            os.path.join(RESULTS_DIR, "extraction.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.results = c
                break

    if not args.results or not os.path.exists(args.results):
        logger.error("Aucun fichier de résultats trouvé. Spécifier --results.")
        sys.exit(1)

    analyze(args.results, args.label)


if __name__ == "__main__":
    main()
