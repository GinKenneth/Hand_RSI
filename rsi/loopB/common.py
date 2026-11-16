"""Loop B shared pieces: the OFFICIAL ACE code
(rsi_paths.ACE_OFFICIAL, the same code that captures and evaluates), LoRA injection, MANO, projection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import rsi_paths  # noqa: E402
ACE_APP = rsi_paths.ACE_OFFICIAL
RELEASE_CKPT = Path(rsi_paths.ACE_CKPT)
RELEASE_OPT = ACE_APP / "options" / "ace_ego_hand_k.yml"
CACHE = S / "loopB" / "cache"                 # latents + per-frame GT per training sequence (model independent)
TAP = 15
EXEC_BLOCKS = 16
WIN = 22                                      # latent window = 85 video frames
GRID = 32
TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
           "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")
HAND_M = 0.15
PREFIX = {"hot3d": "h3__", "arctic": "ar__", "hoi4d": "hd__"}          # sequence-id prefix per dataset


def dataset_of(sid: str) -> str:
    return next(ds for ds, p in PREFIX.items() if sid.startswith(p))


def sid_in(name: str, split_file) -> str:
    """a split-file entry -> its sequence id (HOT3D split files list bare recording names)"""
    ds = Path(split_file).name.split("_", 1)[0]
    return name if name.startswith(PREFIX[ds]) else PREFIX[ds] + name

sys.path.insert(0, str(ACE_APP)); sys.path.insert(0, str(ACE_APP / "third_party"))


def ace_imports():
    os.chdir(ACE_APP)                          # the option paths are repo-relative
    from ace_ego_hand.models.geodit_model import GeoDitModel
    from ace_ego_hand.inference import predict_video
    from ace_ego_hand.mano_utils import mano_forward_batch_full
    return GeoDitModel, predict_video, mano_forward_batch_full


def load_ace(device, checkpoint=RELEASE_CKPT, options=RELEASE_OPT):
    import yaml
    GeoDitModel, _, _ = ace_imports()
    opt = yaml.safe_load(open(options))
    opt["backbone"]["low_cpu_mem_usage"] = True
    model = GeoDitModel(opt, device)
    model.load_inference(str(checkpoint))
    for p in model.net.parameters():
        p.requires_grad_(False)
    return model, opt


def add_domain_adapter(dit, rank: int, alpha: float, blocks: int = EXEC_BLOCKS):
    """Zero-initialised LoRA 'domain' on attention + FFN of the executed DiT blocks (folds into the checkpoint)."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    pattern = r"blocks\.(" + "|".join(map(str, range(blocks))) + r")\.(" + "|".join(t.replace(".", r"\.") for t in TARGETS) + r")$"
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=pattern), dit, adapter_name="domain")
    for m in dit.modules():
        if isinstance(m, LoraLayer) and "domain" in m.lora_A:
            m.set_adapter(["default", "domain"])
    dit.requires_grad_(False)
    a_params, b_params = [], []
    for n, p in dit.named_parameters():
        if ".domain." in n:
            p.data = p.data.float(); p.requires_grad_(True)
            (a_params if ".lora_A." in n else b_params).append(p)
    return a_params, b_params


def project(x, K):
    z = x[..., 2:3].clamp(min=0.05)
    return torch.cat([K[0, 0] * x[..., 0:1] / z + K[0, 2], K[1, 1] * x[..., 1:2] / z + K[1, 2]], -1)


def safe_norm(x, eps=1e-12):
    return torch.sqrt((x * x).sum(-1) + eps)


def huber(x, d):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def mano_joints(go, hp, betas, trans, models, side):
    """Camera-frame 21 joints (OpenPose order), ACE's own MANO forward (flat_hand_mean=False)."""
    _, _, fwd = ace_imports()
    n = hp.shape[0]
    j, _ = fwd(go.reshape(n, 3, 3), hp.reshape(n, 15, 3, 3), betas.reshape(n, 10), models[bool(side)])
    return j.reshape(n, 21, 3) + trans.reshape(n, 1, 3)
