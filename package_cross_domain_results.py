from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


RESULTS_DIR = Path("results")
SRC = RESULTS_DIR / "cross_dataset_bloom_transfer.json"


def _selected_metrics(block: Dict[str, object]) -> Dict[str, object]:
    return dict(block["selected_metrics"])


def _ordinal_distribution_from_confusion(confusion: List[List[int]]) -> Dict[str, int]:
    buckets = {"distance_0": 0, "distance_1": 0, "distance_2plus": 0}
    for i, row in enumerate(confusion):
        for j, count in enumerate(row):
            d = abs(i - j)
            if d == 0:
                buckets["distance_0"] += int(count)
            elif d == 1:
                buckets["distance_1"] += int(count)
            else:
                buckets["distance_2plus"] += int(count)
    return buckets


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    confusions: Dict[str, object] = {}
    ordinal_distributions: Dict[str, object] = {}
    degradation: Dict[str, object] = {}

    for scheme, block in data["schemes"].items():
        confusions[scheme] = {}
        ordinal_distributions[scheme] = {}
        entries = {
            "figshare_to_moocradar": block["figshare_to_moocradar"],
            "moocradar_to_figshare": block["moocradar_to_figshare"],
            "within_dataset_figshare": block["within_dataset_figshare"],
            "within_dataset_moocradar": block["within_dataset_moocradar"],
        }
        for name, entry in entries.items():
            metrics = _selected_metrics(entry)
            confusions[scheme][name] = {
                "selected_model": entry["selected_model"],
                "labels": block["labels"],
                "confusion_matrix": metrics["confusion_matrix"],
                "top_misclassifications": metrics["top_misclassifications"],
            }
            ordinal_distributions[scheme][name] = {
                "selected_model": entry["selected_model"],
                "labels": block["labels"],
                "distance_counts": _ordinal_distribution_from_confusion(metrics["confusion_matrix"]),
                "mean_ordinal_error": metrics["mean_ordinal_error"],
                "within_one_level_accuracy": metrics["within_one_level_accuracy"],
                "severe_error_rate": metrics["severe_error_rate"],
            }

        fig_in = _selected_metrics(block["within_dataset_figshare"])
        mooc_in = _selected_metrics(block["within_dataset_moocradar"])
        fig_to_mooc = _selected_metrics(block["figshare_to_moocradar"])
        mooc_to_fig = _selected_metrics(block["moocradar_to_figshare"])
        degradation[scheme] = {
            "figshare_to_moocradar": {
                "macro_f1_drop_vs_figshare_in_domain": fig_in["macro_f1"] - fig_to_mooc["macro_f1"],
                "within_one_level_drop_vs_figshare_in_domain": fig_in["within_one_level_accuracy"] - fig_to_mooc["within_one_level_accuracy"],
                "severe_error_increase_vs_figshare_in_domain": fig_to_mooc["severe_error_rate"] - fig_in["severe_error_rate"],
            },
            "moocradar_to_figshare": {
                "macro_f1_drop_vs_moocradar_in_domain": mooc_in["macro_f1"] - mooc_to_fig["macro_f1"],
                "within_one_level_drop_vs_moocradar_in_domain": mooc_in["within_one_level_accuracy"] - mooc_to_fig["within_one_level_accuracy"],
                "severe_error_increase_vs_moocradar_in_domain": mooc_to_fig["severe_error_rate"] - mooc_in["severe_error_rate"],
            },
        }

    (RESULTS_DIR / "cross_dataset_confusion_matrices.json").write_text(
        json.dumps(confusions, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "cross_dataset_ordinal_distributions.json").write_text(
        json.dumps(ordinal_distributions, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "cross_dataset_transfer_degradation_summary.json").write_text(
        json.dumps(degradation, indent=2), encoding="utf-8"
    )
    print("packaged")


if __name__ == "__main__":
    main()
