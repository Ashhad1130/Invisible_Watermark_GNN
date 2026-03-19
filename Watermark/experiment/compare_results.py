"""Compare baseline vs optimized vs multi_ring results.
Prints a three-way table, saves CSV + bar-chart plots."""
import json, csv, sys
from pathlib import Path


def load_results(d):
    p = d/"all_attacks_results.json"
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)


def fmt(v, f=".4f"):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:{f}}"
    return str(v)


def _winner(values, higher_is_better):
    """Return label of winning approach or 'TIE'. values: dict {label: float|None}."""
    valid = {k: v for k, v in values.items() if v is not None}
    if len(valid) < 2:
        return "N/A"
    best_val = max(valid.values()) if higher_is_better else min(valid.values())
    winners = [k for k, v in valid.items() if abs(v - best_val) < 1e-4]
    if len(winners) == len(valid):
        return "TIE"
    return "/".join(winners)


def compare(scale="small"):
    base = Path(__file__).resolve().parent/"results"

    # Load whichever result sets exist
    all_approaches = {
        "baseline":   load_results(base/f"{scale}_baseline"),
        "optimized":  load_results(base/f"{scale}_optimized"),
        "multi_ring": load_results(base/f"{scale}_multi_ring"),
    }
    approaches = {k: v for k, v in all_approaches.items() if v}

    if not approaches:
        print(f"No results found for scale='{scale}'. Run experiments first.")
        return

    ap_labels = list(approaches.keys())
    col_w = 10  # column width for each approach

    # Header
    header_sep = "=" * (22 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 10 + 2)
    print(f"\n{header_sep}")
    print(f"  COMPARISON: {' vs '.join(ap_labels).upper()}  ({scale.upper()} SCALE)")
    print(header_sep)

    col_header = " | ".join(f"{lbl:>{col_w}}" for lbl in ap_labels)
    print(f"\n{'Attack':<22} | {'Metric':<16} | {col_header} | {'Winner':>10}")
    print("-" * (22 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 10 + 2))

    attacks = sorted(set(atk for res in approaches.values() for atk in res.keys()))
    metrics_cfg = [
        ("auc",            "AUC",        True),
        ("acc",            "Accuracy",   True),
        ("tpr_at_1fpr",    "TPR@1%FPR",  True),
        ("mean_w_metric",  "W Metric",   False),
        ("clip_score_w_mean", "CLIP (W)", True),
    ]
    win_counts = {lbl: 0 for lbl in ap_labels}
    ties = 0

    for atk in attacks:
        for mk, mn, hb in metrics_cfg:
            row_vals = {lbl: approaches[lbl].get(atk, {}).get(mk) for lbl in ap_labels}
            if all(v is None for v in row_vals.values()):
                continue
            cols = " | ".join(f"{fmt(row_vals[lbl]):>{col_w}}" for lbl in ap_labels)
            w = _winner(row_vals, hb)
            print(f"{atk:<22} | {mn:<16} | {cols} | {w:>10}")
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
            print(f"{atk:<22} | {'Time (s)':<16} | {t_cols} | {'---':>10}")
        print("-" * (22 + 3 + 16 + 3 + (col_w + 3) * len(ap_labels) + 10 + 2))

    # Summary
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for lbl in ap_labels:
        print(f"  {lbl:<12} wins: {win_counts[lbl]}")
    print(f"  {'Ties':<12}:     {ties}")

    # Config info
    print()
    for lbl in ap_labels:
        sample = next(iter(approaches[lbl].values()), {})
        wp = sample.get("watermark_params", {})
        opt = sample.get("optimizations", {})
        info = (f"ch={wp.get('w_channel')}, pattern={wp.get('w_pattern')}, "
                f"mask={wp.get('w_mask_shape','circle')}, r={wp.get('w_radius')}")
        if "w_radius_inner" in wp:
            info += f", r_inner={wp['w_radius_inner']}"
        if opt:
            info += f", steps={opt.get('embed_steps')}/{opt.get('detect_steps')}"
        print(f"  {lbl:<12}: {info}")

    # CSV
    csv_path = base/f"{scale}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Attack", "Approach", "AUC", "Accuracy", "TPR@1%FPR",
                    "Mean_NoW", "Mean_W", "CLIP_NoW", "CLIP_W", "Time_s"])
        for atk in attacks:
            for lbl in ap_labels:
                r = approaches[lbl].get(atk, {})
                if not r: continue
                w.writerow([atk, lbl,
                    fmt(r.get("auc")), fmt(r.get("acc")), fmt(r.get("tpr_at_1fpr")),
                    fmt(r.get("mean_no_w_metric")), fmt(r.get("mean_w_metric")),
                    fmt(r.get("clip_score_mean")), fmt(r.get("clip_score_w_mean")),
                    fmt(r.get("elapsed_seconds"), ".1f")])
    print(f"\n  CSV: {csv_path}")

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pd = base/f"{scale}_plots"; pd.mkdir(parents=True, exist_ok=True)

        # Color palette — one per approach
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        ap_colors = {lbl: colors[i % len(colors)] for i, lbl in enumerate(ap_labels)}

        bar_w = 0.8 / len(ap_labels)
        x = range(len(attacks))

        for mk, mn in [("auc", "AUC"), ("acc", "Accuracy"), ("tpr_at_1fpr", "TPR@1%FPR")]:
            fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 1.8), 6))
            for j, lbl in enumerate(ap_labels):
                vals = [approaches[lbl].get(atk, {}).get(mk, 0) or 0 for atk in attacks]
                offset = (j - len(ap_labels) / 2 + 0.5) * bar_w
                ax.bar([i + offset for i in x], vals, bar_w,
                       label=lbl, color=ap_colors[lbl])
            ax.set_ylabel(mn)
            ax.set_title(f"{mn}: {' vs '.join(ap_labels)}")
            ax.set_xticks(list(x)); ax.set_xticklabels(attacks, rotation=45, ha="right")
            ax.legend(); ax.set_ylim(0, 1.05); plt.tight_layout()
            plt.savefig(pd/f"comparison_{mk}.png", dpi=150); plt.close()
        print(f"  Plots: {pd}")
    except ImportError:
        print("  matplotlib not available, skipping plots.")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="small")
    a = p.parse_args()
    if a.scale == "both":
        compare("small"); compare("large")
    else:
        compare(a.scale)
