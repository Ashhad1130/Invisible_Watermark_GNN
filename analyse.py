"""
Results analysis and visualisation for Tree-Ring Watermark experiments.

Reads ``results/evaluation_results.json`` (produced by ``evaluate.py``)
and outputs:
  - A formatted comparison table to stdout
  - results/figures/heatmap.png         – balanced accuracy heatmap
                                          (methods × attacks)
  - results/figures/rotation_comparison.png – TPR vs rotation angle for
                                              all methods
  - results/figures/delta_plot.png      – Δ score (wm_score − clean_score)
                                          per method, no-attack only

Usage
-----
    python analyse.py [--results_path results/evaluation_results.json]
                      [--output_dir results/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _get_metric(results: list[dict], metric: str) -> tuple[list[str], list[str], np.ndarray]:
    """Extract a metric matrix (methods × attacks).

    Returns
    -------
    method_names : list[str]
    attack_names : list[str]
    matrix       : np.ndarray  shape (n_methods, n_attacks)
    """
    method_names = [r["method"] for r in results]
    attack_names = list(results[0]["attacks"].keys())
    matrix = np.zeros((len(method_names), len(attack_names)))
    for i, result in enumerate(results):
        for j, attack in enumerate(attack_names):
            matrix[i, j] = result["attacks"][attack].get(metric, 0.0)
    return method_names, attack_names, matrix


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """Print a human-readable table of balanced accuracy per method × attack."""
    method_names, attack_names, matrix = _get_metric(results, "balanced_accuracy")

    col_w = 14
    header = f"{'Method':<14}" + "".join(f"{a:<{col_w}}" for a in attack_names)
    sep = "-" * len(header)

    print("\n=== Balanced Accuracy (method × attack) ===")
    print(sep)
    print(header)
    print(sep)
    for i, method in enumerate(method_names):
        row = f"{method:<14}" + "".join(f"{matrix[i, j]:<{col_w}.3f}" for j in range(len(attack_names)))
        print(row)
    print(sep)

    # Average column
    averages = matrix.mean(axis=1)
    print("\n=== Average Balanced Accuracy ===")
    for method, avg in zip(method_names, averages):
        print(f"  {method:<14}: {avg:.3f}")
    print()


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def save_heatmap(results: list[dict], output_path: Path) -> None:
    """Save a balanced-accuracy heatmap (methods × attacks)."""
    method_names, attack_names, matrix = _get_metric(results, "balanced_accuracy")

    fig, ax = plt.subplots(figsize=(max(10, len(attack_names) * 0.9), 4))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Balanced Accuracy")

    ax.set_xticks(range(len(attack_names)))
    ax.set_xticklabels(attack_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_title("Watermark Detection – Balanced Accuracy (method × attack)")

    # Annotate cells
    for i in range(len(method_names)):
        for j in range(len(attack_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Rotation comparison
# ---------------------------------------------------------------------------

def save_rotation_comparison(results: list[dict], output_path: Path) -> None:
    """Line plot of TPR vs rotation angle for each method."""
    rotation_attacks = {
        0: "none",
        15: "rotate_15",
        45: "rotate_45",
        90: "rotate_90",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D"]

    for idx, result in enumerate(results):
        method = result["method"]
        angles = []
        tprs = []
        for angle, attack_key in sorted(rotation_attacks.items()):
            if attack_key in result["attacks"]:
                angles.append(angle)
                tprs.append(result["attacks"][attack_key].get("tpr", 0.0))
        ax.plot(angles, tprs, marker=markers[idx % len(markers)], label=method)

    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TPR vs Rotation Attack")
    ax.set_xticks(list(rotation_attacks.keys()))
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Delta score plot
# ---------------------------------------------------------------------------

def save_delta_plot(results: list[dict], output_path: Path) -> None:
    """Bar chart of Δ score (wm − clean) for the no-attack condition."""
    methods = []
    deltas = []
    for result in results:
        attack_data = result["attacks"].get("none", {})
        delta = attack_data.get("mean_score_wm", 0.0) - attack_data.get("mean_score_clean", 0.0)
        methods.append(result["method"])
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(methods, deltas, color=colors[: len(methods)])

    ax.set_ylabel("Δ Score  (watermarked − clean)")
    ax.set_title("Watermark Score Margin (no attack)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.1, max(0.5, max(deltas) + 0.1))

    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 0.01, 0.01),
            f"{delta:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse watermark evaluation results")
    parser.add_argument(
        "--results_path",
        default="results/evaluation_results.json",
        help="Path to the JSON results file",
    )
    parser.add_argument(
        "--output_dir",
        default="results/figures",
        help="Directory where figures are saved",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", flush=True)
        print("Run evaluate.py first to generate results.", flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(str(results_path))

    print_comparison_table(results)
    save_heatmap(results, output_dir / "heatmap.png")
    save_rotation_comparison(results, output_dir / "rotation_comparison.png")
    save_delta_plot(results, output_dir / "delta_plot.png")

    print("\nAll figures saved to", output_dir)


if __name__ == "__main__":
    main()
