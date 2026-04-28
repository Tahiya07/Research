from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results/cross_dataset_bloom_transfer.json")
FIG_DIR = Path("figures")


def _load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _rows(data: dict) -> list[dict]:
    rows = []
    for scheme in ["binary", "ternary"]:
        block = data["schemes"][scheme]
        for name in [
            "within_dataset_figshare",
            "within_dataset_moocradar",
            "figshare_to_moocradar",
            "moocradar_to_figshare",
        ]:
            m = block[name]["selected_metrics"]
            rows.append(
                {
                    "scheme": scheme,
                    "setting": name,
                    "model": block[name]["selected_model"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "mean_ordinal_error": m["mean_ordinal_error"],
                    "within_one_level_accuracy": m["within_one_level_accuracy"],
                    "severe_error_rate": m["severe_error_rate"],
                }
            )
    return rows


def _save_performance_table(df: pd.DataFrame) -> None:
    out_csv = FIG_DIR / "cross_domain_performance_table.csv"
    df.to_csv(out_csv, index=False)

    disp = df.copy()
    for col in ["accuracy", "macro_f1", "mean_ordinal_error", "within_one_level_accuracy", "severe_error_rate"]:
        disp[col] = disp[col].map(lambda x: f"{x:.3f}")
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title("Cross-Domain Bloom Performance", fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_domain_performance_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(data: dict) -> None:
    block = data["schemes"]["ternary"]
    pairs = [
        ("figshare_to_moocradar", "Figshare -> MoocRadar"),
        ("moocradar_to_figshare", "MoocRadar -> Figshare"),
    ]
    labels = block["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (key, title) in zip(axes, pairs):
        cm = np.array(block[key]["selected_metrics"]["confusion_matrix"], dtype=float)
        row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
        norm = cm / row_sums
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
    fig.suptitle("Cross-Domain Transfer Heatmaps (Ternary Collapse Space)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "transfer_heatmaps_ternary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_ordinal_shift(data: dict) -> None:
    block = data["schemes"]["ternary"]
    labels = [
        "Fig in-domain",
        "Mooc in-domain",
        "Fig -> Mooc",
        "Mooc -> Fig",
    ]
    keys = [
        "within_dataset_figshare",
        "within_dataset_moocradar",
        "figshare_to_moocradar",
        "moocradar_to_figshare",
    ]
    buckets = ["distance_0", "distance_1", "distance_2plus"]
    counts = []
    for key in keys:
        cm = np.array(block[key]["selected_metrics"]["confusion_matrix"])
        dist = {"distance_0": 0, "distance_1": 0, "distance_2plus": 0}
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                d = abs(i - j)
                if d == 0:
                    dist["distance_0"] += int(cm[i, j])
                elif d == 1:
                    dist["distance_1"] += int(cm[i, j])
                else:
                    dist["distance_2plus"] += int(cm[i, j])
        total = max(1, sum(dist.values()))
        counts.append([dist[b] / total for b in buckets])
    arr = np.array(counts)
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(labels))
    colors = ["#1f77b4", "#ffbf00", "#d62728"]
    pretty = ["Exact level", "Within one level", "Severe (2+)"]
    for i, bucket in enumerate(buckets):
        ax.bar(labels, arr[:, i], bottom=bottom, label=pretty[i], color=colors[i])
        bottom += arr[:, i]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of predictions")
    ax.set_title("Ordinal Error Shift Across Domains (Ternary)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ordinal_error_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_key_figure(df: pd.DataFrame) -> None:
    sub = df[df["scheme"] == "ternary"].copy()
    order = [
        "within_dataset_figshare",
        "within_dataset_moocradar",
        "figshare_to_moocradar",
        "moocradar_to_figshare",
    ]
    labels = ["Fig in-domain", "Mooc in-domain", "Fig -> Mooc", "Mooc -> Fig"]
    sub["order"] = sub["setting"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("order")
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, sub["macro_f1"], width=w, label="Macro-F1", color="#4c72b0")
    ax.bar(x + w / 2, sub["within_one_level_accuracy"], width=w, label="Within-one-level acc.", color="#55a868")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Domain Shift Preserves Ordinal Structure Despite Classification Collapse")
    ax.legend()
    for xi, v in zip(x - w / 2, sub["macro_f1"]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, sub["within_one_level_accuracy"]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "domain_shift_preserves_ordinal_structure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    df = pd.DataFrame(_rows(data))
    _save_performance_table(df)
    _plot_heatmaps(data)
    _plot_ordinal_shift(data)
    _plot_key_figure(df)
    print("figures-generated")


if __name__ == "__main__":
    main()
