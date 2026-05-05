"""Generate report charts for the latest tuned insurance claim model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
MODEL_FILE = ROOT / "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = ROOT / "best_threshold.json"
TRAIN_REPORT_FILE = ROOT / "tuned_model_report.json"
THRESHOLD_SWEEP_FILE = ROOT / "threshold_sweep.csv"
HIGH_CONF_FILE = ROOT / "high_confidence_report.json"
CONFUSION_FILE = ROOT / "confusion_matrix_tuned.csv"


def set_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#0f172a",
            "savefig.facecolor": "#0f172a",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "axes.titlecolor": "#f8fafc",
            "font.size": 10,
        }
    )


def ensure_dir() -> None:
    ASSET_DIR.mkdir(exist_ok=True)


def load_model():
    return joblib.load(MODEL_FILE)


def plot_confusion_matrix() -> None:
    set_dark_theme()
    df = pd.read_csv(CONFUSION_FILE, index_col=0)
    cm = df.values

    plt.figure(figsize=(6.4, 5.4), dpi=180)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.color_palette(["#0f172a", "#f97316"], as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="#1f2937",
        square=True,
        annot_kws={"color": "white", "fontsize": 13, "fontweight": "bold"},
    )
    ax.set_title("Confusion Matrix - Tuned Model", pad=16, fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.set_xticklabels(["NO CLAIM", "CLAIM"], rotation=0)
    ax.set_yticklabels(["NO CLAIM", "CLAIM"], rotation=0)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "confusion_matrix_tuned.png", bbox_inches="tight", facecolor="#0f172a")
    plt.close()


def plot_threshold_curve() -> None:
    set_dark_theme()
    df = pd.read_csv(THRESHOLD_SWEEP_FILE)
    plt.figure(figsize=(9, 5.5), dpi=180)
    plt.plot(df["threshold"], df["precision"], label="Precision", color="#38bdf8", linewidth=2.2)
    plt.plot(df["threshold"], df["recall"], label="Recall", color="#f97316", linewidth=2.2)
    plt.plot(df["threshold"], df["f1"], label="F1", color="#a78bfa", linewidth=2.2)
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        selected_threshold = float(json.load(f)["threshold"])
    plt.axvline(selected_threshold, color="#f8fafc", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.text(selected_threshold + 0.005, 0.1, f"Selected {selected_threshold:.3f}", color="#f8fafc", fontsize=9)
    plt.title("Threshold Sweep on Validation Set", fontsize=15, fontweight="bold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.18)
    ax = plt.gca()
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "threshold_sweep_curve.png", bbox_inches="tight", facecolor="#0f172a")
    plt.close()


def plot_top_x_curve() -> None:
    set_dark_theme()
    with open(HIGH_CONF_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = report["top_x_results"]
    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5.5), dpi=180)
    x = df["coverage"] * 100.0
    plt.plot(x, df["precision"], label="Precision", color="#22c55e", linewidth=2.2)
    plt.plot(x, df["recall"], label="Recall", color="#f59e0b", linewidth=2.2)
    plt.plot(x, df["f1"], label="F1", color="#60a5fa", linewidth=2.2)
    for _, row in df.iterrows():
        plt.scatter(row["coverage"] * 100.0, row["precision"], color="#22c55e", s=24)
    plt.title("Top-X% High-Confidence Strategy", fontsize=15, fontweight="bold")
    plt.xlabel("Coverage (% of rows accepted)")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.18)
    ax = plt.gca()
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "top_x_curve.png", bbox_inches="tight", facecolor="#0f172a")
    plt.close()


def plot_feature_importance() -> None:
    set_dark_theme()
    model = load_model()
    pre = model.named_steps["preprocess"]
    booster = model.named_steps["model"]

    feature_names = list(pre.get_feature_names_out())
    importances = np.asarray(booster.feature_importances_).reshape(-1)
    if len(feature_names) < len(importances):
        feature_names = feature_names + [f"padded_feature_{i+1}" for i in range(len(importances) - len(feature_names))]
    elif len(feature_names) > len(importances):
        feature_names = feature_names[: len(importances)]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(15).iloc[::-1]

    plt.figure(figsize=(9, 6.5), dpi=180)
    plt.barh(df["feature"], df["importance"], color="#fb7185")
    plt.title("Top Feature Importances - XGBoost", fontsize=15, fontweight="bold")
    plt.xlabel("Importance")
    ax = plt.gca()
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "feature_importance_tuned.png", bbox_inches="tight", facecolor="#0f172a")
    plt.close()


def main() -> None:
    ensure_dir()
    plot_confusion_matrix()
    plot_threshold_curve()
    plot_top_x_curve()
    plot_feature_importance()
    print(f"Saved charts to {ASSET_DIR}")


if __name__ == "__main__":
    main()
