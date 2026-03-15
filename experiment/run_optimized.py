"""
Optimized runner. Imports from original repo without modification.
Optimizations:
  1. Multi-channel watermarking (w_channel=-1): all 4 latent channels
  2. Larger radius (w_radius=14): more frequency coefficients
  3. More inversion steps (test_steps=75): better DDIM inversion accuracy
  4. Amplitude-scaled injection (1.1x): stronger watermark energy
  5. Ensemble per-channel detection: robust to channel-specific attacks
"""
import sys, os, json, time, copy, argparse
from pathlib import Path
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import torch, numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent / "tree-ring-watermark"
sys.path.insert(0, str(REPO_ROOT))
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import (set_random_seed, transform_img, get_watermarking_mask,
                         image_distortion, measure_similarity, circle_mask)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_compat import get_device, get_watermarking_pattern, eval_watermark as _eval_base
from landscape_prompts import get_prompt
from configs import ExperimentConfig, AttackConfig

# Optimization #4: Amplitude-scaled injection
def inject_watermark_scaled(init_latents_w, watermarking_mask, gt_patch, args, scale_factor=1.1):
    orig_dtype = init_latents_w.dtype
    init_latents_w = init_latents_w.float()
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    scaled_patch = gt_patch.clone()
    scaled_patch[watermarking_mask] = gt_patch[watermarking_mask] * scale_factor
    if args.w_injection == "complex":
        init_latents_w_fft[watermarking_mask] = scaled_patch[watermarking_mask]
    elif args.w_injection == "seed":
        init_latents_w[watermarking_mask] = scaled_patch[watermarking_mask]
        return init_latents_w.to(orig_dtype)
    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
    return init_latents_w.to(orig_dtype)

# Optimization #5: Ensemble detection
def eval_watermark_ensemble(rev_no_w, rev_w, mask, gt_patch, args):
    if args.w_channel != -1:
        return _eval_base(rev_no_w, rev_w, mask, gt_patch, args)
    rev_no_fft = torch.fft.fftshift(torch.fft.fft2(rev_no_w.float()), dim=(-1, -2))
    rev_w_fft = torch.fft.fftshift(torch.fft.fft2(rev_w.float()), dim=(-1, -2))
    no_w_ch, w_ch = [], []
    for ch in range(rev_no_w.shape[1]):
        cm = mask[:, ch:ch+1]
        if not cm.any(): continue
        no_w_ch.append(torch.abs(rev_no_fft[:, ch:ch+1][cm] - gt_patch[:, ch:ch+1][cm]).mean().item())
        w_ch.append(torch.abs(rev_w_fft[:, ch:ch+1][cm] - gt_patch[:, ch:ch+1][cm]).mean().item())
    if not w_ch: return _eval_base(rev_no_w, rev_w, mask, gt_patch, args)
    return float(np.mean(no_w_ch)), float(np.mean(w_ch))

def build_args(config, attack):
    a = argparse.Namespace()
    a.model_id=config.model_id; a.image_length=config.image_length; a.num_images=config.num_images
    a.guidance_scale=config.guidance_scale; a.num_inference_steps=config.num_inference_steps
    a.test_num_inference_steps=config.test_num_inference_steps; a.gen_seed=config.gen_seed
    a.start=config.start; a.end=config.end; a.run_name=f"{config.name}_{attack.name}"
    a.with_tracking=False; a.max_num_log_image=100
    wm=config.watermark
    a.w_seed=wm.w_seed; a.w_channel=wm.w_channel; a.w_pattern=wm.w_pattern
    a.w_mask_shape=wm.w_mask_shape; a.w_radius=wm.w_radius; a.w_measurement=wm.w_measurement
    a.w_injection=wm.w_injection; a.w_pattern_const=wm.w_pattern_const
    a.r_degree=attack.r_degree; a.jpeg_ratio=attack.jpeg_ratio; a.crop_scale=attack.crop_scale
    a.crop_ratio=attack.crop_ratio; a.gaussian_blur_r=attack.gaussian_blur_r
    a.gaussian_std=attack.gaussian_std; a.brightness_factor=attack.brightness_factor; a.rand_aug=attack.rand_aug
    a.dataset="Gustavosta/Stable-Diffusion-Prompts"
    a.reference_model=config.reference_model; a.reference_model_pretrain=config.reference_model_pretrain
    return a

def load_ckpt(p):
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

def save_ckpt(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,"w") as f: json.dump(data, f, indent=2)

def run_optimized(config, skip_clip=False, scale_factor=1.1):
    device = get_device()

    print(f"\n{'='*60}\nOPTIMIZED: {config.name} | Device: {device} | Images: {config.start}-{config.end}")
    print(f"Watermark: ch={config.watermark.w_channel}, pattern={config.watermark.w_pattern}, r={config.watermark.w_radius}")
    print(f"Opts: scale={scale_factor}, inv_steps={config.test_num_inference_steps}\n{'='*60}\n")

    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir/config.checkpoint_dir/config.name
    res_dir = base_dir/config.results_dir/config.name
    out_dir = base_dir/config.outputs_dir/config.name
    for d in [ckpt_dir, res_dir, out_dir]: d.mkdir(parents=True, exist_ok=True)

    args = build_args(config, config.attacks[0] if config.attacks else AttackConfig())

    print("Loading Stable Diffusion pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(args.model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention(); print("  xformers enabled")
    except: pass

    ref_model=ref_clip_preprocess=ref_tokenizer=None
    if not skip_clip and config.reference_model:
        print(f"Loading CLIP: {config.reference_model}...")
        import open_clip
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(config.reference_model, pretrained=config.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(config.reference_model)

    text_embeddings = pipe.get_text_embedding("")
    gt_patch = get_watermarking_pattern(pipe, args, device)

    all_results = {}
    for attack in config.attacks:
        args_a = build_args(config, attack)
        ckpt_path = ckpt_dir/f"{attack.name}_checkpoint.json"
        ckpt = load_ckpt(ckpt_path)

        if ckpt and ckpt.get("completed"):
            print(f"  {attack.name}: already done, loading.")
            all_results[attack.name] = ckpt["final_results"]; continue

        start_idx=config.start; results=[]; no_w_metrics=[]; w_metrics=[]; clips=[]; clips_w=[]; saved=[]
        if ckpt and not ckpt.get("completed"):
            start_idx=ckpt["last_idx"]+1; results=ckpt.get("results",[]); no_w_metrics=ckpt.get("no_w_metrics",[])
            w_metrics=ckpt.get("w_metrics",[]); clips=ckpt.get("clips",[]); clips_w=ckpt.get("clips_w",[]); saved=ckpt.get("saved",[])
            print(f"  {attack.name}: resuming from {start_idx}")

        t0=time.time()
        for i in tqdm(range(start_idx, config.end), desc=f"  {attack.name}"):
            seed=i+config.gen_seed; prompt=get_prompt(i)
            set_random_seed(seed)
            lat_no_w=pipe.get_random_latents()
            out_no_w=pipe(prompt,num_images_per_prompt=1,guidance_scale=args_a.guidance_scale,num_inference_steps=args_a.num_inference_steps,height=args_a.image_length,width=args_a.image_length,latents=lat_no_w)
            img_no_w=out_no_w.images[0]

            lat_w=copy.deepcopy(lat_no_w)
            mask=get_watermarking_mask(lat_w, args_a, device)
            lat_w=inject_watermark_scaled(lat_w, mask, gt_patch, args_a, scale_factor=scale_factor)
            out_w=pipe(prompt,num_images_per_prompt=1,guidance_scale=args_a.guidance_scale,num_inference_steps=args_a.num_inference_steps,height=args_a.image_length,width=args_a.image_length,latents=lat_w)
            img_w=out_w.images[0]

            if len(saved)<5:
                d=out_dir/attack.name; d.mkdir(parents=True,exist_ok=True)
                img_no_w.save(d/f"img_{i:04d}_no_wm.png"); img_w.save(d/f"img_{i:04d}_wm.png"); saved.append(i)

            img_no_w_a, img_w_a = image_distortion(img_no_w, img_w, seed, args_a)

            t_no=transform_img(img_no_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_no=pipe.forward_diffusion(latents=pipe.get_image_latents(t_no,sample=False),text_embeddings=text_embeddings,guidance_scale=1,num_inference_steps=args_a.test_num_inference_steps)
            t_w=transform_img(img_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_w=pipe.forward_diffusion(latents=pipe.get_image_latents(t_w,sample=False),text_embeddings=text_embeddings,guidance_scale=1,num_inference_steps=args_a.test_num_inference_steps)

            nm,wm_=eval_watermark_ensemble(rev_no,rev_w,mask,gt_patch,args_a)
            cs_no=cs_w=0.0
            if ref_model:
                sims=measure_similarity([img_no_w,img_w],prompt,ref_model,ref_clip_preprocess,ref_tokenizer,device)
                cs_no=sims[0].item(); cs_w=sims[1].item()

            results.append({"i":i,"prompt":prompt,"no_w":nm,"w":wm_,"clip_no":cs_no,"clip_w":cs_w})
            no_w_metrics.append(-nm); w_metrics.append(-wm_); clips.append(cs_no); clips_w.append(cs_w)

            if (i-config.start+1)%config.checkpoint_every==0 or i==config.end-1:
                save_ckpt(ckpt_path,{"attack":attack.name,"last_idx":i,"completed":False,"results":results,
                    "no_w_metrics":no_w_metrics,"w_metrics":w_metrics,"clips":clips,"clips_w":clips_w,"saved":saved})

        elapsed=time.time()-t0
        preds=no_w_metrics+w_metrics; labels=[0]*len(no_w_metrics)+[1]*len(w_metrics)
        fpr,tpr,_=metrics.roc_curve(labels,preds,pos_label=1)
        auc=metrics.auc(fpr,tpr); acc=np.max(1-(fpr+(1-tpr))/2)
        low=tpr[np.where(fpr<.01)[0][-1]] if len(np.where(fpr<.01)[0])>0 else 0.0

        final={"attack":attack.name,"approach":"optimized","num_images":config.end-config.start,
            "auc":float(auc),"acc":float(acc),"tpr_at_1fpr":float(low),
            "mean_no_w_metric":float(mean([-m for m in no_w_metrics])),"mean_w_metric":float(mean([-m for m in w_metrics])),
            "clip_score_mean":float(mean(clips)) if clips and any(c!=0 for c in clips) else None,
            "clip_score_w_mean":float(mean(clips_w)) if clips_w and any(c!=0 for c in clips_w) else None,
            "elapsed_seconds":elapsed,
            "watermark_params":{"w_channel":config.watermark.w_channel,"w_pattern":config.watermark.w_pattern,"w_radius":config.watermark.w_radius},
            "optimizations":{"scale_factor":scale_factor,"test_num_inference_steps":config.test_num_inference_steps,
                "multi_channel":config.watermark.w_channel==-1,"ensemble_detection":config.watermark.w_channel==-1}}
        all_results[attack.name]=final
        save_ckpt(ckpt_path,{"attack":attack.name,"last_idx":config.end-1,"completed":True,"results":results,
            "no_w_metrics":no_w_metrics,"w_metrics":w_metrics,"clips":clips,"clips_w":clips_w,"saved":saved,"final_results":final})
        print(f"  AUC:{auc:.4f} Acc:{acc:.4f} TPR@1%FPR:{low:.4f} Time:{elapsed:.1f}s")

    with open(res_dir/"all_attacks_results.json","w") as f: json.dump(all_results,f,indent=2)
    print(f"\nOptimized results: {res_dir/'all_attacks_results.json'}")
    return all_results

if __name__=="__main__":
    import argparse as ap; p=ap.ArgumentParser(); p.add_argument("--scale",default="small"); p.add_argument("--skip_clip",action="store_true"); p.add_argument("--scale_factor",type=float,default=1.1); a=p.parse_args()
    from configs import get_small_scale_optimized, get_large_scale_optimized
    run_optimized(get_small_scale_optimized() if a.scale=="small" else get_large_scale_optimized(), skip_clip=a.skip_clip, scale_factor=a.scale_factor)
