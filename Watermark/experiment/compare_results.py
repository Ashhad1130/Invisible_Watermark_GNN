"""Compare baseline vs optimized vs multi_ring vs multi_ring_allchan results.
Prints a comparison table, saves CSV + publication-quality bar charts.

Usage:
  python compare_results.py --scale large --dataset category
  python compare_results.py --scale large --dataset gustavosta
"""
import json, csv, sys
from pathlib import Path

# Approximate paper reference values (Wen et al. 2023, Tree-Ring ring key,
# SD 2.1, Gustavosta dataset, 100 images). Marked as approximate.
PAPER_REF_AUC = {
    "no_attack":           1.000,
    "rotation_75":         0.990,
    "jpeg_25":             0.990,
    "crop_0.75":           0.970,
    "gaussian_blur_8":     1.000,
    "gaussian_noise_0.1":  0.980,
    "brightness_6":        1.000,
}


def load_results(d):
    p = d / "all_attacks_results.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt(v, f=".4f"):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:{f}}"
    return str(v)


def _winner(values, higher_is_better=True):
    valid = {k: v for k, v in values.items() if v is not None}
    if len(valid) < 2:
        return "N/A"
    best_val = max(valid.values()) if higher_is_better else min(valid.values())
    winners = [k for k, v in valid.items() if abs(v - best_val) < 1e-4]
    return "TIE" if len(winners) == len(valid) else "/".join(winners)


def compare(scale="small", dataset_suffix=""):
    """
    scale: "small" or "large"
    dataset_suffix: "" for gustavosta, "category" for category prompts, "landscape" for landscape
    """
    base = Path(__file__).resolve().parent / "results"
    suf  = f"_{dataset_suffix}" if dataset_suffix else ""

    all_approaches = {
        "baseline":     load_results(base / f"{scale}_baseline{suf}"),
        "optimized":    load_results(base / f"{scale}_optimized{suf}"),
        "multi_ring":   load_results(base / f"{scale}_multi_ring{suf}"),
        "mr_allchan":   load_results(base / f"{scale}_multi_ring_allchan{suf}"),
    }
    approaches = {k: v for k, v in all_approaches.items() if v}

    if not approaches:
        label = f"scale='{scale}'" + (f", dataset='{dataset_suffix}'" if dataset_suffix else "")
        print(f"No results found for {label}. Run experiments first.")
        return

    ap_labels = list(approaches.keys())
    col_w = 11

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "=" * (24 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 12)
    ds_label = dataset_suffix.upper() if dataset_suffix else "GUSTAVOSTA"
    print(f"\n{sep}")
    print(f"  RESULTS: {' vs '.join(ap_labels).upper()}  |  {scale.upper()} SCALE  |  {ds_label}")
    print(sep)

    col_header = " | ".join(f"{lbl:>{col_w}}" for lbl in ap_labels)
    print(f"\n{'Attack':<24} | {'Metric':<16} | {col_header} | {'Winner':>12}")
    print("-" * (24 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 12))

    attacks = sorted(set(atk for res in approaches.values() for atk in res.keys()))
    metrics_cfg = [
        ("auc",              "AUC",         True),
        ("acc",              "Accuracy",    True),
        ("tpr_at_1fpr",      "TPR@1%FPR",   True),
        ("mean_w_metric",    "W Metric",    False),
        ("clip_score_w_mean","CLIP (W)",    True),
    ]
    win_counts = {lbl: 0 for lbl in ap_labels}
    ties = 0
    problem_attacks = []  # attacks where best AUC < 0.97

    for atk in attacks:
        for mk, mn, hb in metrics_cfg:
            row_vals = {lbl: approaches[lbl].get(atk, {}).get(mk) for lbl in ap_labels}
            if all(v is None for v in row_vals.values()):
                continue
            cols = " | ".join(f"{fmt(row_vals[lbl]):>{col_w}}" for lbl in ap_labels)
            w = _winner(row_vals, hb)

            # Flag paper reference gap for AUC
            paper_flag = ""
            if mk == "auc" and atk in PAPER_REF_AUC:
                best_ours = max((v for v in row_vals.values() if v is not None), default=0)
                gap = PAPER_REF_AUC[atk] - best_ours
                if gap > 0.02:
                    paper_flag = f"  ^paper={PAPER_REF_AUC[atk]:.3f}"
                    problem_attacks.append(atk)

            print(f"{atk:<24} | {mn:<16} | {cols} | {w:>12}{paper_flag}")

            if mk == "auc":
                if w == "TIE":
                    ties += 1
                elif w != "N/A":
                    for lbl in w.split("/"):
                        if lbl in win_counts:
                            win_counts[lbl] += 1

        # Time row
        times = {lbl: approaches[lbl].get(atk, {}).get("elapsed_seconds") for lbl in ap_labels}
        if any(v is not None for v in times.values()):
            t_cols = " | ".join(
                f"{times[lbl]:>{col_w}.1f}" if times[lbl] is not None else f"{'N/A':>{col_w}}"
                for lbl in ap_labels)
            print(f"{atk:<24} | {'Time (s)':<16} | {t_cols} | {'---':>12}")
        print("-" * (24 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 12))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  SUMMARY (AUC wins)\n{'='*60}")
    for lbl in ap_labels:
        print(f"  {lbl:<16} wins: {win_counts[lbl]}")
    print(f"  {'Ties':<16}:     {ties}")

    # ── Gaps vs paper ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  GAPS vs PAPER (Wen et al. 2023, SD 2.1, approximate)\n{'='*60}")
    for atk in attacks:
        if atk not in PAPER_REF_AUC:
            continue
        best_ours = max(
            (approaches[lbl].get(atk, {}).get("auc") or 0 for lbl in ap_labels),
            default=0)
        gap = PAPER_REF_AUC[atk] - best_ours
        flag = "  [PROBLEM]" if gap > 0.02 else ("  [OK]" if gap <= 0.005 else "")
        print(f"  {atk:<24}: paper={PAPER_REF_AUC[atk]:.3f}  ours_best={best_ours:.4f}  gap={gap:+.4f}{flag}")

    # ── Config info ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  CONFIG\n{'='*60}")
    for lbl in ap_labels:
        sample = next(iter(approaches[lbl].values()), {})
        wp  = sample.get("watermark_params", {})
        opt = sample.get("optimizations", {})
        n   = sample.get("num_images", "?")
        info = (f"n={n}, ch={wp.get('w_channel')}, pattern={wp.get('w_pattern')}, "
                f"mask={wp.get('w_mask_shape', 'circle')}, r={wp.get('w_radius')}")
        if "w_radius_inner" in wp:
            info += f", r_inner={wp['w_radius_inner']}"
        if opt:
            info += f", steps={opt.get('embed_steps')}/{opt.get('detect_steps')}"
        print(f"  {lbl:<16}: {info}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = base / f"{scale}{suf}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Attack", "Approach", "AUC", "Accuracy", "TPR@1%FPR",
                    "Mean_NoW", "Mean_W", "CLIP_NoW", "CLIP_W", "Time_s", "Paper_AUC"])
        for atk in attacks:
            for lbl in ap_labels:
                r = approaches[lbl].get(atk, {})
                if not r:
                    continue
                w.writerow([atk, lbl,
                    fmt(r.get("auc")), fmt(r.get("acc")), fmt(r.get("tpr_at_1fpr")),
                    fmt(r.get("mean_no_w_metric")), fmt(r.get("mean_w_metric")),
                    fmt(r.get("clip_score_mean")), fmt(r.get("clip_score_w_mean")),
                    fmt(r.get("elapsed_seconds"), ".1f"),
                    fmt(PAPER_REF_AUC.get(atk))])
    print(f"\n  CSV saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        pd = base / f"{scale}{suf}_plots"
        pd.mkdir(parents=True, exist_ok=True)

        colors = {
            "baseline":   "#4C72B0",
            "optimized":  "#DD8452",
            "multi_ring": "#55A868",
            "mr_allchan": "#C44E52",
        }
        default_colors = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

        for i, lbl in enumerate(ap_labels):
            if lbl not in colors:
                colors[lbl] = default_colors[i % len(default_colors)]

        n_ap   = len(ap_labels)
        bar_w  = 0.7 / max(n_ap, 1)
        x      = np.arange(len(attacks))
        atk_labels = [a.replace("gaussian_", "g_").replace("brightness_", "br_")
                        .replace("rotation_", "rot_").replace("crop_", "crop_")
                        .replace("no_attack", "clean") for a in attacks]

        for mk, mn, y_label in [
            ("auc",         "AUC",        "AUC"),
            ("tpr_at_1fpr", "TPR@1%FPR",  "TPR at 1% FPR"),
            ("acc",         "Accuracy",   "Accuracy"),
        ]:
            fig, ax = plt.subplots(figsize=(max(12, len(attacks) * 1.5), 6))

            for j, lbl in enumerate(ap_labels):
                vals = [approaches[lbl].get(atk, {}).get(mk) or 0 for atk in attacks]
                offset = (j - n_ap / 2 + 0.5) * bar_w
                ax.bar(x + offset, vals, bar_w, label=lbl, color=colors[lbl],
                       alpha=0.85, edgecolor="white", linewidth=0.5)

            # Paper reference line (AUC only, basic attacks only)
            if mk == "auc":
                ref_vals = [PAPER_REF_AUC.get(atk) for atk in attacks]
                if any(v is not None for v in ref_vals):
                    ref_y = [v if v is not None else float("nan") for v in ref_vals]
                    ax.plot(x, ref_y, "k--", linewidth=1.5, label="Paper (approx.)",
                            marker="D", markersize=5, zorder=5)

            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f"{mn} — {scale.upper()} scale, {ds_label}\n"
                         f"({' vs '.join(ap_labels)})", fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(atk_labels, rotation=40, ha="right", fontsize=9)
            ax.set_ylim(0, 1.08)
            ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
            ax.legend(fontsize=9, loc="lower left")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(pd / f"{mk}.png", dpi=180)
            plt.close()

        # ── Heatmap: AUC across all attacks × approaches ──────────────────────
        try:
            import pandas as pd_lib
            auc_data = {lbl: [approaches[lbl].get(atk, {}).get("auc") or 0
                              for atk in attacks]
                        for lbl in ap_labels}
            df = pd_lib.DataFrame(auc_data, index=atk_labels)

            fig, ax = plt.subplots(figsize=(max(6, n_ap * 2), max(5, len(attacks) * 0.6)))
            im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)
            ax.set_xticks(range(n_ap)); ax.set_xticklabels(ap_labels, fontsize=10)
            ax.set_yticks(range(len(attacks))); ax.set_yticklabels(atk_labels, fontsize=9)
            for i in range(len(attacks)):
                for j in range(n_ap):
                    v = df.values[i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="black" if v > 0.85 else "white")
            plt.colorbar(im, ax=ax, label="AUC")
            ax.set_title(f"AUC Heatmap — {scale.upper()} / {ds_label}", fontsize=12)
            plt.tight_layout()
            plt.savefig(pd / "heatmap_auc.png", dpi=180)
            plt.close()
        except ImportError:
            pass  # pandas optional

        print(f"  Plots saved: {pd}")

    except ImportError:
        print("  matplotlib not available — skipping plots.")

    print()
    return approaches


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale",   default="large", choices=["small", "large"])
    p.add_argument("--dataset", default="category",
                   choices=["gustavosta", "category", "landscape"])
    a = p.parse_args()
    suf = "" if a.dataset == "gustavosta" else a.dataset
    compare(a.scale, suf)
