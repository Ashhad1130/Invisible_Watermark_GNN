"""Compare baseline vs optimized results. Prints table, saves CSV + plots."""
import json, csv, sys
from pathlib import Path
from typing import Dict

def load_results(d):
    p = d/"all_attacks_results.json"
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

def fmt(v, f=".4f"):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:{f}}"
    return str(v)

def compare(scale="small"):
    base = Path(__file__).resolve().parent/"results"
    bl = load_results(base/f"{scale}_baseline")
    op = load_results(base/f"{scale}_optimized")
    if not bl and not op: print(f"No results for {scale}. Run experiments first."); return

    print(f"\n{'='*100}\n  COMPARISON: BASELINE vs OPTIMIZED  ({scale.upper()} SCALE)\n{'='*100}")
    print(f"\n{'Attack':<22} | {'Metric':<16} | {'Baseline':>10} | {'Optimized':>10} | {'Delta':>10} | {'Winner':>8}")
    print("-"*100)

    attacks = sorted(set(list(bl.keys())+list(op.keys())))
    ms = [("auc","AUC",True),("acc","Accuracy",True),("tpr_at_1fpr","TPR@1%FPR",True),
          ("mean_w_metric","W Metric",False),("clip_score_w_mean","CLIP (W)",True)]
    summary = {"opt":0,"base":0,"tie":0}

    for atk in attacks:
        b,o = bl.get(atk,{}), op.get(atk,{})
        for mk,mn,hb in ms:
            bv,ov = b.get(mk), o.get(mk)
            if bv is None and ov is None: continue
            bs,os_ = fmt(bv), fmt(ov)
            if bv is not None and ov is not None:
                d=ov-bv; ds=f"{d:+.4f}"
                w="OPT" if (d>0.001 if hb else d<-0.001) else ("BASE" if (d<-0.001 if hb else d>0.001) else "TIE")
                summary["opt" if w=="OPT" else "base" if w=="BASE" else "tie"]+=1
            else: ds="N/A"; w="N/A"
            print(f"{atk:<22} | {mn:<16} | {bs:>10} | {os_:>10} | {ds:>10} | {w:>8}")
        bt,ot = b.get("elapsed_seconds"), o.get("elapsed_seconds")
        if bt and ot: print(f"{atk:<22} | {'Time (s)':<16} | {bt:>10.1f} | {ot:>10.1f} | {bt/ot if ot else 0:>9.2f}x | {'---':>8}")
        print("-"*100)

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  Optimized wins: {summary['opt']}\n  Baseline wins:  {summary['base']}\n  Ties:           {summary['tie']}")
    if bl:
        p=next(iter(bl.values())).get("watermark_params",{})
        print(f"\n  Baseline:  ch={p.get('w_channel')}, pattern={p.get('w_pattern')}, radius={p.get('w_radius')}")
    if op:
        p=next(iter(op.values())).get("watermark_params",{}); o_=next(iter(op.values())).get("optimizations",{})
        print(f"  Optimized: ch={p.get('w_channel')}, pattern={p.get('w_pattern')}, radius={p.get('w_radius')}")
        print(f"  Opts:      scale={o_.get('scale_factor')}, inv_steps={o_.get('test_num_inference_steps')}, multi_ch={o_.get('multi_channel')}")

    # CSV
    csv_path = base/f"{scale}_comparison.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Attack","Approach","AUC","Accuracy","TPR@1%FPR","Mean_NoW","Mean_W","CLIP_NoW","CLIP_W","Time_s"])
        for atk in attacks:
            for ap_,res in [("baseline",bl),("optimized",op)]:
                r=res.get(atk,{});
                if not r: continue
                w.writerow([atk,ap_,fmt(r.get("auc")),fmt(r.get("acc")),fmt(r.get("tpr_at_1fpr")),
                    fmt(r.get("mean_no_w_metric")),fmt(r.get("mean_w_metric")),
                    fmt(r.get("clip_score_mean")),fmt(r.get("clip_score_w_mean")),fmt(r.get("elapsed_seconds"),".1f")])
    print(f"\n  CSV: {csv_path}")

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pd = base/f"{scale}_plots"; pd.mkdir(parents=True, exist_ok=True)
        for mk,mn in [("auc","AUC"),("acc","Accuracy"),("tpr_at_1fpr","TPR@1%FPR")]:
            bv=[bl.get(a,{}).get(mk,0) or 0 for a in attacks]
            ov=[op.get(a,{}).get(mk,0) or 0 for a in attacks]
            x=range(len(attacks)); w_=0.35
            fig,ax=plt.subplots(figsize=(max(10,len(attacks)*1.5),6))
            ax.bar([i-w_/2 for i in x],bv,w_,label="Baseline",color="#4C72B0")
            ax.bar([i+w_/2 for i in x],ov,w_,label="Optimized",color="#DD8452")
            ax.set_ylabel(mn); ax.set_title(f"{mn}: Baseline vs Optimized")
            ax.set_xticks(list(x)); ax.set_xticklabels(attacks,rotation=45,ha="right")
            ax.legend(); ax.set_ylim(0,1.05); plt.tight_layout()
            plt.savefig(pd/f"comparison_{mk}.png",dpi=150); plt.close()
        print(f"  Plots: {pd}")
    except ImportError: print("  matplotlib not available, skipping plots.")
    print()

if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--scale",default="small"); a=p.parse_args()
    if a.scale=="both": compare("small"); compare("large")
    else: compare(a.scale)
