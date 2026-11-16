"""Fold a trained student into a deployable ACE checkpoint (no inference code change).

LoRA: the base checkpoint's 'default' adapter (release: rank 64; theta_{k-1}: rank 64+32(k-1)) and
the new 'domain' adapter (rank r, scale alpha/r) are concatenated into one LoRA with alpha = rank,
i.e. scale 1, with the domain scale folded into B:
    A' = [A_default; A_domain],  B' = [B_default * s_d, B_domain * s_dom]
Head: the projector tensors are replaced by the trained (EMA by default) ones.
The output directory holds ace_ego_hand_k.pt + ace_ego_hand_k.yml, usable as
ACE_CHECKPOINT / ACE_OPTIONS by the unchanged delivery.

  python -m distill.merge_distill --student runs/T/step1000.pt --out_dir models/S0prime [--verify]
  python -m distill.merge_distill --student runs/r2_TpNY/best --out_dir models/theta2 --verify
      ("<run>/best" = the saved step with the lowest D_hold score in <run>/hold_eval.jsonl)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS in verify()
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopB import common as C


def pick_best(run_dir: Path, key: str = "common_epe_mano") -> Path:   # the MANO path, not the direct-2D head
    """Saved step with the best D_hold score (step 0 = the base itself is not a candidate)."""
    rows = [json.loads(l) for l in (run_dir / "hold_eval.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["step"] > 0 and (run_dir / f"step{r['step']}.pt").is_file()]
    best = min(rows, key=lambda r: r[key])
    print("best step", json.dumps({k: best[k] for k in ("step", "common_epe_direct", "common_epe_mano", "missed_hands")}),
          flush=True)
    return run_dir / f"step{best['step']}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--raw", action="store_true", help="use raw instead of EMA weights")
    ap.add_argument("--verify", action="store_true", help="check merged == adapter model on one window")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    args.student, args.out_dir = args.student.resolve(), args.out_dir.resolve()   # ACE import chdirs
    if args.student.name == "best":
        args.student = pick_best(args.student.parent)
    st = torch.load(args.student, map_location="cpu", weights_only=False)
    base_ckpt = Path(st.get("base_ckpt") or C.RELEASE_CKPT)
    base_options = Path(st.get("base_options") or C.RELEASE_OPT)
    domain = st["domain"] if args.raw else st["ema_domain"]
    head = st["head"] if args.raw else st["ema_head"]
    r, scale_dom = int(st["rank"]), float(st["alpha"]) / int(st["rank"])
    opt = yaml.safe_load(open(base_options))
    base_rank = int(opt["backbone"]["lora"]["lora_rank"])
    scale_def = float(opt["backbone"]["lora"]["lora_alpha"]) / base_rank
    dom = {k.replace(".domain.", ".default."): v for k, v in domain.items()}
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    bt = ck["backbone_trainable"]
    missing = [k for k in dom if k not in bt]
    if missing:
        raise KeyError(f"adapter keys absent from release checkpoint: {missing[:3]}")
    merged = {}
    for k, v in bt.items():
        if ".lora_A." in k:
            add = dom.get(k, torch.zeros(r, v.shape[1]))
            merged[k] = torch.cat([v, add.to(v.dtype)], 0)
        elif ".lora_B." in k:
            add = dom.get(k, torch.zeros(v.shape[0], r)) * scale_dom
            merged[k] = torch.cat([(v.float() * scale_def).to(v.dtype), add.to(v.dtype)], 1)
        else:
            merged[k] = v
    ck["backbone_trainable"] = merged
    if head is not None:
        proj = dict(ck["projectors"])
        unknown = [k for k in head if k not in proj]
        if unknown:
            raise KeyError(f"head keys absent from release projectors: {unknown[:3]}")
        for k, v in head.items():
            proj[k] = v.to(proj[k].dtype)
        ck["projectors"] = proj
    history = list(ck.get("rsi_history", [])) + ([ck["rsi_distill"]] if "rsi_distill" in ck else [])
    ck["rsi_history"] = history
    ck["rsi_distill"] = dict(student=str(args.student), teacher=st.get("teacher"), step=st.get("step"),
                             round=st.get("round"), base_ckpt=str(base_ckpt),
                             weights="raw" if args.raw else "ema", domain_rank=r, base_rank=base_rank,
                             head_replaced=head is not None, args=st.get("args"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out_dir / "ace_ego_hand_k.pt")
    opt["backbone"]["lora"]["lora_rank"] = base_rank + r
    opt["backbone"]["lora"]["lora_alpha"] = base_rank + r
    for key in ("model_root", "videox_config", "caption_embed"):
        opt["paths"][key] = str((C.ACE_APP / opt["paths"][key]).resolve())
    (args.out_dir / "ace_ego_hand_k.yml").write_text(yaml.safe_dump(opt, sort_keys=False))
    info = dict(rank=base_rank + r, domain_tensors=len(dom), head_replaced=head is not None,
                out=str(args.out_dir))
    print(json.dumps(info), flush=True)
    if args.verify:
        verify(args, st, domain, head, base_ckpt, base_options)


def _fp32(model):
    """Run the backbone and head in fp32: under the deployed bf16 autocast a small adapter's effect is
    the size of bf16 rounding, which differs between the split and the concatenated LoRA (measured
    bf16 max|diff| cam_trans 7e-3 vs fp32 4e-7 for the same weights)."""
    model.net.backbone.dit.float()
    model.net.backbone.compute_dtype = torch.float32
    model.net.projectors.float()
    return model


def _deterministic():
    """Blackwell GPUs, torch 2.7: the default attention kernels are not deterministic. The same model on the same
    window differs run to run by up to 0.02 (cam_trans, exists_2d), which hides any merge error.
    With deterministic algorithms, math SDPA and no TF32 two runs are bit-identical. Used for the check only."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


@torch.no_grad()
def verify(args, st, domain, head, base_ckpt, base_options):
    """Adapter model (base + domain LoRA + head) vs merged checkpoint on one cached window, in fp32, deterministic."""
    import numpy as np
    _deterministic()
    device = torch.device("cuda", args.gpu)
    d = next(p for p in sorted(C.CACHE.iterdir()) if (p / "latents.pt").is_file())
    meta = json.loads((d / "meta.json").read_text())
    lat = torch.load(d / "latents.pt")[:, :C.WIN].unsqueeze(0).float().to(device)
    n_video = 4 * (C.WIN - 1) + 1
    model, _ = C.load_ace(device, base_ckpt, base_options)
    _fp32(model)
    a_p, b_p = C.add_domain_adapter(model.net.backbone.dit, int(st["rank"]), float(st["alpha"]))
    own = dict(model.net.backbone.dit.named_parameters())
    for k, v in domain.items():
        own[k].data.copy_(v.to(own[k].dtype))
    if head is not None:
        ph = dict(model.net.projectors.named_parameters())
        for k, v in head.items():
            ph[k].data.copy_(v.to(ph[k].dtype))
    model.net.backbone.dit.to(device)
    ref = model.net.forward_emode(lat, n_video, [meta["intr"]])["preds"][C.TAP][0]
    del model
    torch.cuda.empty_cache()
    import yaml as _y
    opt = _y.safe_load(open(args.out_dir / "ace_ego_hand_k.yml"))
    opt["backbone"]["low_cpu_mem_usage"] = True
    GeoDitModel, *_ = C.ace_imports()
    merged = GeoDitModel(opt, device)
    merged.load_inference(str(args.out_dir / "ace_ego_hand_k.pt"))
    _fp32(merged)
    out = merged.net.forward_emode(lat, n_video, [meta["intr"]])["preds"][C.TAP][0]
    diffs = {k: float((out[k].float() - ref[k].float()).abs().max()) for k in ("cam_trans", "direct_joints2d", "exists_2d")}
    print("VERIFY fp32 max|merged-adapter|", json.dumps(diffs), flush=True)
    if diffs["direct_joints2d"] > 1e-4 or diffs["cam_trans"] > 1e-4 or diffs["exists_2d"] > 1e-3:
        raise SystemExit("merged checkpoint does not reproduce the adapter model")


if __name__ == "__main__":
    main()
