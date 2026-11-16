"""Loop B: distil the round's teacher (θ_{k-1} + H_k*) into θ_k, with the 3D GT of the training split.

  torchrun --nproc_per_node N -m loopB.train --base-ckpt CK --base-opt YML --teacher R1 --out runs/R1 [--steps 2000]
     [--w_teacher 1.0] [--w_gt 1.0] [--seqs a,b]          (--w_teacher 0 = control C1: GT fine-tuning only)

Per 85-frame window of a training sequence (loopB/cache/<sid>: latents, gt3d.npz, teacher_<name>.npz), per hand:
  GT term (annotated hand-frames): 2D of the in-frame joints (MANO projection + direct 2D head), wrist-relative 3D,
    absolute wrist position, global orientation; presence: exists_3d -> 1, exists_2d -> any GT joint in frame
  teacher term (hand-frames the teacher claims, exists > 0.5, and -- default --teacher_where nogt --
    WITHOUT GT for that hand; e.g. HOT3D's 62 recordings without public GT): 2D (MANO projection vs the teacher's 3D
    projected, direct 2D head vs the teacher's 2D), wrist-relative 3D, absolute wrist, global orientation, articulation
  presence where there is no GT: both heads -> the teacher's existence
Student = θ_{k-1} + a zero-initialised rank-r LoRA on the executed DiT blocks + the projector head; LoRA+, EMA, cosine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopB import common as C  # noqa: E402


class Seq:
    def __init__(self, sid: str, teacher: str | None):
        d = C.CACHE / sid
        self.name = sid
        self.meta = json.loads((d / "meta.json").read_text())
        self.lat = torch.load(d / "latents.pt", mmap=True)
        self.n, self.f_lat = self.meta["frames"], self.lat.shape[1]
        self.intr = self.meta["intr"]
        self.K = np.asarray(self.meta["K"], np.float32)
        self.wh = np.asarray([self.meta["W"], self.meta["H"]], np.float32)
        self.g = dict(np.load(d / "gt3d.npz"))
        self.t = dict(np.load(d / f"teacher_{teacher}.npz")) if teacher else None


def span_of(uv):
    return (uv.max(-2).values - uv.min(-2).values).norm(dim=-1).clamp(min=40.0)


_HEAD_INIT = None      # the projector heads' values at the start of training (the round's base network)


def base_forward(net, lat, nf, intr):
    """outputs of the network as it was before this training (θ_{k-1}): the LoRA 'domain' branch scaled to 0 and the
    heads put back to their initial values, no grad. Called BEFORE the window's own forward: the swap must not touch
    tensors that the backward pass of the main forward will read."""
    from peft.tuners.lora import LoraLayer
    layers = [m for m in net.backbone.dit.modules() if isinstance(m, LoraLayer) and "domain" in m.scaling]
    scal = [m.scaling["domain"] for m in layers]
    heads = list(net.projectors.parameters())
    cur = [h.data.clone() for h in heads]
    try:
        for m in layers:
            m.scaling["domain"] = 0.0
        for h, h0 in zip(heads, _HEAD_INIT):
            h.data.copy_(h0)
        with torch.no_grad():
            s0 = net.forward_emode(lat, nf, [intr])["preds"][C.TAP][0]
    finally:
        for m, sc in zip(layers, scal):
            m.scaling["domain"] = sc
        for h, c in zip(heads, cur):
            h.data.copy_(c)
    return {k: x.detach() for k, x in s0.items() if torch.is_tensor(x)}


def temporal_term(js, J, mask, scale=0.005):
    """velocity + acceleration matching of camera-space joints over consecutive frames where mask holds (TCMR,
    Choi et al. CVPR 2021). The evaluator's Jitter is the second difference of predicted joints inside one 81-frame
    segment = one training window, so it is learnable here; per-frame losses barely see a few mm of frame-to-frame
    noise. scale 5 mm, Huber knee at 1 scale."""
    m1 = mask[1:] & mask[:-1]
    m2 = mask[2:] & mask[1:-1] & mask[:-2]
    lv = C.huber(C.safe_norm((js[1:] - js[:-1]) - (J[1:] - J[:-1])) / scale, 1.0).mean(-1)
    la = C.huber(C.safe_norm((js[2:] - 2 * js[1:-1] + js[:-2]) - (J[2:] - 2 * J[1:-1] + J[:-2])) / scale, 1.0).mean(-1)
    return (torch.where(m1, lv, 0.0).sum() / m1.float().sum().clamp(min=1.0)
            + torch.where(m2, la, 0.0).sum() / m2.float().sum().clamp(min=1.0))


def lowpass(x, k):
    """moving average over frames (dim 0), k odd; edges averaged over the frames that exist"""
    if k <= 1:
        return x
    f = x.reshape(x.shape[0], -1).t().unsqueeze(0)
    y = F.avg_pool1d(f, k, stride=1, padding=k // 2, count_include_pad=False)
    return y.squeeze(0).t().reshape(x.shape)


def window_loss(net, models, v: Seq, args, device):
    a = random.randint(0, v.f_lat - C.WIN)
    frames = np.arange(4 * a, 4 * a + 4 * (C.WIN - 1) + 1)
    valid = frames < v.n
    fr = frames.clip(max=v.n - 1)
    lat = v.lat[:, a:a + C.WIN].unsqueeze(0).float().to(device)
    if args.latent_noise > 0:
        lat = lat + args.latent_noise * lat.std() * torch.randn_like(lat)
    anchor = args.w_anchor > 0 and C.dataset_of(v.name) in args.anchor_datasets.split(',')
    s0 = base_forward(net, lat, len(frames), v.intr) if anchor else None
    s = net.forward_emode(lat, len(frames), [v.intr])["preds"][C.TAP][0]
    K = torch.from_numpy(v.K).to(device); wh = torch.from_numpy(v.wh).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    stats, total, tot_t = {}, 0.0, 0.0            # total = GT group (GT terms + anchor), tot_t = teacher group
    # --teacher_all_datasets: where the teacher beats the GT-tuned network on EVERY column (ARCTIC: ACE already
    # fits its GT, the harness adds smoothing / 2D / placement), the teacher supervises all frames with all terms
    tall = C.dataset_of(v.name) in [x for x in args.teacher_all_datasets.split(',') if x]
    w_gt_eff = args.w_gt_all if (tall and args.w_gt_all >= 0) else args.w_gt
    w_t_eff = args.w_teacher_all if tall else args.w_teacher
    t_all_terms = tall or args.teacher_terms == 'all'
    T = lambda arr: torch.from_numpy(np.nan_to_num(np.ascontiguousarray(arr[fr]))).float().to(device)   # noqa: E731
    for side in (0, 1):
        go_s = s["go"][:, side].reshape(-1, 3, 3); hp_s = s["hp"][:, side].reshape(-1, 15, 3, 3)
        js = C.mano_joints(go_s, hp_s, s["betas"][:, side], s["cam_trans"][:, side], models, side)
        ps = C.project(js, K)
        dir_px = s["direct_joints2d"][:, side] * wh
        e3 = s["exists_3d"][:, side].clamp(1e-4, 1 - 1e-4); e2 = s["exists_2d"][:, side].clamp(1e-4, 1 - 1e-4)

        def inner(J, uv):
            """per frame: every joint in front of the camera and inside the image with an args.edge_px margin"""
            m = args.edge_px
            return ((J[..., 2] > 0.01) & (uv[..., 0] >= m) & (uv[..., 0] < wh[0] - m) & (uv[..., 1] >= m)
                    & (uv[..., 1] < wh[1] - m)).all(-1)

        def pose_terms(J, uv_t, R, mask, vis, hp_t=None, weight=None, placement=True, go=False):
            span = span_of(uv_t)
            nv = vis.float().sum(-1).clamp(min=1.0)
            l_uv = torch.where(vis, C.huber(C.safe_norm(ps - uv_t) / span[:, None], 0.05), 0.0).sum(-1) / nv
            if not placement and args.teacher_rel_frame == "hand":
                # Loop C R1 P1: wrist-relative joints in camera coordinates still carry the global orientation, so the
                # 'articulation' teacher kept supervising GO; compare them in each hand's own frame (R_go^T (J - J0))
                rel_s = torch.einsum("fji,fkj->fki", go_s, js - js[:, :1])
                rel_t = torch.einsum("fji,fkj->fki", R, J - J[:, :1])
                l_rel = C.huber(C.safe_norm(rel_s - rel_t) / C.HAND_M, 0.05).mean(-1)
            else:
                l_rel = C.huber(C.safe_norm((js - js[:, :1]) - (J - J[:, :1])) / C.HAND_M, 0.05).mean(-1)
            l_root = C.huber(C.safe_norm(js[:, 0] - J[:, 0]) / C.HAND_M, 0.05)
            l_go = ((go_s - R) ** 2).sum((-1, -2))
            # placement = 2D reprojection, absolute wrist, global orientation; without it only the hand's own pose
            # (wrist-relative 3D joints + articulation) is supervised (--teacher_terms articulation)
            per = (l_uv + args.w_rel * l_rel + args.w_root * l_root + args.w_rot * l_go) if placement else args.w_rel * l_rel
            if go and not placement:       # --teacher_go: global orientation only (no 2D, wrist position or direct 2D)
                per = per + args.w_rot * l_go
            if hp_t is not None:
                per = per + args.w_rot * ((hp_s - hp_t) ** 2).sum((-1, -2)).mean(-1)
            mf = mask.float()      # normalised by the frame COUNT: low-quality pseudo-labels contribute less in absolute terms
            return (torch.where(mask, per, 0.0) * (1.0 if weight is None else weight)).sum() / mf.sum().clamp(min=1.0), mf.sum()

        # GT term
        Jg_np = v.g["joints_cam"][fr, side]; Rg_np = v.g["rot"][fr, side]
        mg = torch.from_numpy(v.g["valid"][fr, side] & np.isfinite(Jg_np).all((-1, -2)) & np.isfinite(Rg_np).all((-1, -2))).to(device) & valid_t
        Jg, Rg = T(v.g["joints_cam"])[:, side], T(v.g["rot"])[:, side]
        uvg = C.project(Jg, K)
        visg = (Jg[..., 2] > 0.01) & (uvg[..., 0] >= 0) & (uvg[..., 0] < wh[0]) & (uvg[..., 1] >= 0) & (uvg[..., 1] < wh[1]) & mg[:, None]
        gt_pose, n_gt = pose_terms(Jg, uvg, Rg, mg, visg)
        nvg = visg.float().sum(-1).clamp(min=1.0)
        gt_dir = (torch.where(visg, C.huber(C.safe_norm(dir_px - uvg) / span_of(uvg)[:, None], 0.05), 0.0).sum(-1) / nvg)
        gt_dir = torch.where(mg, gt_dir, 0.0).sum() / mg.float().sum().clamp(min=1.0)
        mgf = mg.float()
        gt_ex = ((F.binary_cross_entropy(e3, torch.ones_like(e3), reduction="none") +
                  F.binary_cross_entropy(e2, visg.any(-1).float(), reduction="none")) * mgf).sum() / mgf.sum().clamp(min=1.0)
        total = total + w_gt_eff * (gt_pose + args.w_dir * gt_dir + args.w_exists * gt_ex)
        if args.w_temporal > 0:
            # --temporal_onscreen: only where the GT hand is in the image (a hand that left the frame must not be
            # held in place by the smoothness term)
            # --edge_px: border slivers dominate HOT3D false positives (true hand just outside the frame, the prediction
            # pokes a few px in, mostly near a hand entering / leaving): smooth only fully-inside hands
            if args.edge_px > 0:
                g_tmp = temporal_term(js, Jg, mg & inner(Jg, uvg))
            else:
                g_tmp = temporal_term(js, Jg, (mg & visg.any(-1)) if args.temporal_onscreen else mg)
            total = total + args.w_temporal * g_tmp
            stats.setdefault("gt_temporal", []).append(float(g_tmp))
        # teacher term
        if v.t is not None and w_t_eff > 0:
            te = v.t["exists"][fr, side]
            Jt_np = v.t["joints_cam"][fr, side]
            mt = torch.from_numpy((te > 0.5) & np.isfinite(Jt_np).all((-1, -2))).to(device) & valid_t
            if args.teacher_where == "nogt" and not tall:   # where GT exists, GT alone supervises
                mt = mt & ~mg
            Jt, Rt, Ht = T(v.t["joints_cam"])[:, side], T(v.t["rot"])[:, side], T(v.t["hp"])[:, side]
            uvt = C.project(Jt, K)
            vist = (Jt[..., 2] > 0.01) & (uvt[..., 0] >= 0) & (uvt[..., 0] < wh[0]) & (uvt[..., 1] >= 0) & (uvt[..., 1] < wh[1]) & mt[:, None]
            # per hand-frame pseudo-label quality weight (loopB/targets.py quality_weight; 1 for older teacher files)
            wq = T(v.t["weight"])[:, side] if "weight" in v.t else torch.ones(len(frames), device=device)
            t_pose, n_t = pose_terms(Jt, uvt, Rt, mt, vist, Ht, weight=wq, placement=t_all_terms,
                                     go=args.teacher_go)
            if args.teacher_terms == 'none' and not tall and C.dataset_of(v.name) not in [x for x in args.teacher_pose_datasets.split(',') if x]:
                # pose from GT only; the teacher keeps presence /
                t_pose = t_pose * 0.0                          # 2D / temporal: the student passes it on articulation
            if args.w_temporal_teacher > 0:
                if args.edge_px > 0:
                    t_tmp = temporal_term(js, Jt, mt & inner(Jt, uvt))
                else:
                    t_tmp = temporal_term(js, Jt, (mt & vist.any(-1)) if args.temporal_onscreen else mt)
                tot_t = tot_t + args.w_temporal_teacher * t_tmp
                stats.setdefault("t_temporal", []).append(float(t_tmp))
            t2d = T(v.t["joints_2d"])[:, side]
            ntv = vist.float().sum(-1).clamp(min=1.0)
            t_dir = torch.where(vist, C.huber(C.safe_norm(dir_px - t2d) / span_of(uvt)[:, None], 0.05), 0.0).sum(-1) / ntv
            t_dir = (torch.where(mt, t_dir, 0.0) * wq).sum() / mt.float().sum().clamp(min=1.0)
            # presence on frames without GT for this hand: follow the teacher
            nog = ~mg & valid_t
            if args.edge_px > 0:          # the teacher's hand cut by / near the border: no presence label
                nog = nog & ~(mt & vist.any(-1) & ~inner(Jt, uvt))
            nog = nog.float()
            tex = torch.from_numpy(np.nan_to_num(te)).float().to(device).clamp(0, 1)
            # --teacher_e2_vis: the 2D head learns "visible", not "exists" (the teacher's off-screen hands taught it
            # to call them visible)
            tex2 = torch.where(mt, tex * vist.any(-1).float(), tex) if args.teacher_e2_vis else tex
            t_ex = ((F.binary_cross_entropy(e3, tex, reduction="none") + F.binary_cross_entropy(e2, tex2, reduction="none")) * nog).sum() \
                / nog.sum().clamp(min=1.0)
            t_place = args.w_dir * t_dir if (t_all_terms or args.teacher_2d) else 0.0     # the direct 2D head is placement
            w_tex = args.w_exists if args.w_teacher_exists < 0 else args.w_teacher_exists
            tot_t = tot_t + w_t_eff * (t_pose + t_place + w_tex * t_ex)
            stats.setdefault("teacher", []).append(float(t_pose))
            stats.setdefault("t_exists", []).append(float(t_ex))
        # the harness's RULES as self-supervised losses on every frame with a hand (GT or teacher presence), no
        # labels needed and camera-agnostic: S2HAND 2D-3D consistency (2103.11703), S2HAND(V) motion / shape consistency
        # (2201.09548). Modules -> losses: anchor_consensus / anchor_fuse -> cons2d; temporal_articulation -> cons3d;
        # shape_prior x4 -> shape; root_trajectory / temporal_consistency -> jerk.
        if args.w_cons2d > 0 or args.w_cons3d > 0 or args.w_shape > 0 or args.w_jerk > 0:
            pres = mg.clone()
            if v.t is not None:
                pres = pres | (torch.from_numpy(np.nan_to_num(v.t["exists"][fr, side]) > 0.5).to(device) & valid_t)
            inimg = (js[..., 2] > 0.01) & (ps[..., 0] >= 0) & (ps[..., 0] < wh[0]) & (ps[..., 1] >= 0) & (ps[..., 1] < wh[1]) & pres[:, None]
            pf = pres.float(); npf = pf.sum().clamp(min=1.0)
            if args.w_cons2d > 0:
                c2 = torch.where(inimg, C.huber(C.safe_norm(ps - dir_px) / span_of(ps)[:, None], 0.05), 0.0).sum(-1) \
                    / inimg.float().sum(-1).clamp(min=1.0)
                c2 = (c2 * pf).sum() / npf
                total = total + args.w_cons2d * c2; stats.setdefault("cons2d", []).append(float(c2))
            if args.w_cons3d > 0:
                dj = s["direct_joints_cam"][:, side]
                c3 = C.huber(C.safe_norm((dj - dj[:, :1]) - (js - js[:, :1])) / C.HAND_M, 0.05).mean(-1) \
                    + C.huber(C.safe_norm(dj[:, 0] - js[:, 0]) / C.HAND_M, 0.05)
                c3 = (c3 * pf).sum() / npf
                total = total + args.w_cons3d * c3; stats.setdefault("cons3d", []).append(float(c3))
            if args.w_shape > 0:
                b = s["betas"][:, side]
                bm = (b * pf[:, None]).sum(0) / npf
                var = (((b - bm) ** 2).sum(-1) * pf).sum() / npf
                vis = (e2.detach() * pf).sum() / npf
                cs = var + (1.0 - vis) * (bm ** 2).sum() / 10.0
                total = total + args.w_shape * cs; stats.setdefault("shape", []).append(float(cs))
            if args.w_jerk > 0:
                on = pres & inimg.any(-1)
                m3 = on[3:] & on[2:-1] & on[1:-2] & on[:-3]
                d3 = js[3:] - 3 * js[2:-1] + 3 * js[1:-2] - js[:-3]
                cj = torch.where(m3, C.huber(C.safe_norm(d3) / 0.005, 1.0).mean(-1), 0.0).sum() / m3.float().sum().clamp(min=1.0)
                total = total + args.w_jerk * cj; stats.setdefault("jerk", []).append(float(cj))
        with torch.no_grad():
            stats.setdefault("gt", []).append(float(gt_pose))
            if mg.any():
                stats.setdefault("gt_mpjpe_mm", []).extend((1000 * ((js - js[:, :1]) - (Jg - Jg[:, :1])).norm(dim=-1).mean(-1))[mg].tolist())
                stats.setdefault("gt_root_mm", []).extend((1000 * (js[:, 0] - Jg[:, 0]).norm(dim=-1))[mg].tolist())
    if s0 is not None:
        # anchor (LwF, 1606.09282; SDFT, 2402.13669): on this dataset's windows, keep the placement outputs (global
        # orientation, camera translation, direct 2D head) of the network as it was before this training
        vm = valid_t.float()[:, None]
        k = args.anchor_lowpass      # > 1: anchor only the slow part, so the temporal term can still smooth
        terms = set(args.anchor_terms.split(','))   # per-output lock: the columns where the teacher is worse
        a = torch.zeros(len(frames), 2, device=device)
        if 'go' in terms:
            a = a + ((lowpass(s["go"], k) - lowpass(s0["go"], k)) ** 2).reshape(len(frames), 2, -1).sum(-1)
        if 'ct' in terms:
            a = a + C.huber(C.safe_norm(lowpass(s["cam_trans"], k) - lowpass(s0["cam_trans"], k)) / C.HAND_M, 0.05)
        if '2d' in terms:
            a = a + C.huber(C.safe_norm((lowpass(s["direct_joints2d"], k) - lowpass(s0["direct_joints2d"], k)) * wh) / 100.0,
                            0.05).mean(-1)
        if 'rel' in terms:           # wrist-relative joints (MPJPE / PA columns), from both networks' MANO outputs
            for side in (0, 1):
                j_s = C.mano_joints(s["go"][:, side].reshape(-1, 3, 3), s["hp"][:, side].reshape(-1, 15, 3, 3),
                                    s["betas"][:, side], s["cam_trans"][:, side], models, side)
                j_0 = C.mano_joints(s0["go"][:, side].reshape(-1, 3, 3), s0["hp"][:, side].reshape(-1, 15, 3, 3),
                                    s0["betas"][:, side], s0["cam_trans"][:, side], models, side)
                r_s, r_0 = lowpass(j_s - j_s[:, :1], k), lowpass(j_0 - j_0[:, :1], k)
                a[:, side] = a[:, side] + C.huber(C.safe_norm(r_s - r_0) / C.HAND_M, 0.05).mean(-1)
        anc = (a * vm).sum() / (2 * vm.sum()).clamp(min=1.0)
        total = total + args.w_anchor * anc
        stats.setdefault("anchor", []).append(float(anc))
    return total, tot_t, stats


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, int(os.environ.get("RSI_GPU", "0"))
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", type=Path, default=C.RELEASE_CKPT)
    ap.add_argument("--base-opt", type=Path, default=C.RELEASE_OPT)
    ap.add_argument("--train-list", default=str(C.S / "data/splits/hot3d_train.txt"),
                    help="comma-separated split files (one per dataset)")
    ap.add_argument("--mix", default="", help="dataset sampling shares, e.g. hot3d=0.57,arctic=0.43 "
                    "(default: proportional to latent frames)")
    ap.add_argument("--teacher", default=None, help="teacher_<name>.npz; none = GT only")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--windows_per_step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loraplus_ratio", type=float, default=4.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--head_lr_mult", type=float, default=0.1)
    ap.add_argument("--latent_noise", type=float, default=0.02)
    ap.add_argument("--w_gt", type=float, default=1.0)
    ap.add_argument("--w_teacher", type=float, default=1.0)
    ap.add_argument("--w_dir", type=float, default=1.0)
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_root", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=0.25)
    ap.add_argument("--w_exists", type=float, default=0.5)
    ap.add_argument("--w_teacher_exists", type=float, default=-1.0,
                    help="weight of the teacher presence term (< 0: --w_exists; added for an ablation proposed by Loop C after round 4)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqs", default="", help="subset (smoke tests)")
    ap.add_argument("--teacher_grad", choices=("plain", "pcgrad"), default="plain",
                    help="pcgrad = remove the component of the teacher gradient that conflicts with the GT (+anchor) gradient")
    ap.add_argument("--w_anchor", type=float, default=0.0, help="weight of the placement anchor to the pre-training network")
    ap.add_argument("--anchor_datasets", default="arctic", help="datasets whose windows get the anchor (comma list)")
    ap.add_argument("--teacher_all_datasets", default="", help="comma list: teacher on every frame with all terms there")
    ap.add_argument("--w_teacher_all", type=float, default=1.0, help="teacher weight on --teacher_all_datasets windows")
    ap.add_argument("--w_gt_all", type=float, default=-1.0, help="GT weight there (< 0: --w_gt)")
    ap.add_argument("--w_temporal", type=float, default=0.0, help="velocity/acceleration matching to GT on GT frames")
    ap.add_argument("--w_temporal_teacher", type=float, default=0.0, help="velocity/acceleration matching to the teacher on teacher frames")
    ap.add_argument("--teacher_go", action="store_true", help="articulation teacher also supervises global orientation")
    ap.add_argument("--anchor_lowpass", type=int, default=0, help="odd k > 1: anchor the k-frame moving average only")
    ap.add_argument("--anchor_terms", default="go,ct,2d", help="anchored outputs: go, ct, 2d, rel (comma list)")
    ap.add_argument("--teacher_rel_frame", choices=("camera", "hand"), default="camera",
                    help="frame of the articulation teacher's wrist-relative joint term (hand = GO-free)")
    ap.add_argument("--teacher_2d", action="store_true", help="teacher also supervises the direct 2D head (EPE2D)")
    ap.add_argument("--temporal_onscreen", action="store_true", help="temporal terms only where the target hand is in the image")
    ap.add_argument("--edge_px", type=int, default=0,
                    help="> 0: border mask -- temporal terms only where the target hand lies fully inside the image with this "
                         "margin, and no teacher presence where the teacher's hand is cut by / near the border")
    ap.add_argument("--teacher_e2_vis", action="store_true",
                    help="teacher presence: exists_2d -> teacher exists AND a teacher joint in the image (default: exists)")
    ap.add_argument("--w_cons2d", type=float, default=0.0, help="direct 2D head vs MANO projection consistency")
    ap.add_argument("--w_cons3d", type=float, default=0.0, help="direct 3D head vs MANO joints consistency")
    ap.add_argument("--w_shape", type=float, default=0.0, help="one shape per window, visibility-weighted prior")
    ap.add_argument("--w_jerk", type=float, default=0.0, help="minimum-jerk prior on the student's own joints")
    ap.add_argument("--teacher_pose_datasets", default="", help="with --teacher_terms none: keep the articulation teacher on these datasets")
    ap.add_argument("--teacher_terms", choices=("articulation", "all", "none"), default="articulation",
                    help="what the teacher supervises: articulation = the hand's own pose only (wrist-relative 3D joints, finger rotations; existence stays); all = also placement (2D, absolute wrist, global orientation, direct 2D head). With all, R1 gave F_k = C1 on HOT3D and -4%% vs C1 on ARCTIC, the loss in CT / EPE2D / GO: the HOT3D pseudo-labels moved the shared network's placement")
    ap.add_argument("--teacher_where", choices=("nogt", "all"), default="nogt",
                    help="hand-frames the teacher term supervises: nogt = only those without GT (on frames "
                         "with GT the teacher's approximate targets conflicted with the GT; in a smoke test C1 beat F_k by 1.9%% on "
                         "HOT3D); all = every hand-frame the teacher claims")
    args = ap.parse_args()
    args.out = args.out.resolve()
    ema_decay = 1.0 - 5.0 / args.steps
    rank, world, device = setup_dist()
    local_windows = args.windows_per_step // world
    random.seed(args.seed * 1000 + rank); np.random.seed(args.seed * 1000 + rank); torch.manual_seed(args.seed)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()} | {"world": world}, indent=1))
    model, _ = C.load_ace(device, args.base_ckpt, args.base_opt)
    net = model.net; dit = net.backbone.dit
    a_params, b_params = C.add_domain_adapter(dit, args.rank, float(args.rank))
    dit.to(device)
    head = []
    for p in net.projectors.parameters():
        p.data = p.data.float(); p.requires_grad_(True); head.append(p)
    groups = [dict(params=a_params, lr=args.lr), dict(params=b_params, lr=args.lr * args.loraplus_ratio),
              dict(params=head, lr=args.lr * args.head_lr_mult)]
    params = a_params + b_params + head
    global _HEAD_INIT
    _HEAD_INIT = [p.detach().clone() for p in head]        # the base network's heads (anchor)
    optim = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99)); base_lrs = [g["lr"] for g in groups]
    ema = {id(p): p.detach().clone().float() for p in params}
    sids = [C.sid_in(s, lst) for lst in args.train_list.split(",") for s in Path(lst).read_text().split()]
    if args.seqs:
        sids = [s for s in sids if s in args.seqs.split(",")]
    seqs = [Seq(s, args.teacher) for s in sids if (C.CACHE / s / "latents.pt").is_file()]
    seqs = [v for v in seqs if v.f_lat >= C.WIN]
    if not args.teacher or args.w_teacher == 0:        # GT-only (control C1): sequences without any GT add nothing
        seqs = [v for v in seqs if v.g["valid"].any()]
    w = np.array([v.f_lat for v in seqs], float)
    ds_of = np.array([C.dataset_of(v.name) for v in seqs])
    if args.mix:                                   # each dataset gets its share; inside it, sequences by length
        mix = {k: float(x) for k, x in (kv.split("=") for kv in args.mix.split(","))}
        for d in set(ds_of):
            m = ds_of == d
            w[m] = w[m] / w[m].sum() * mix.get(d, 0.0)
    w /= w.sum()
    if rank == 0:
        print(json.dumps(dict(sequences=len(seqs), share={d: round(float(w[ds_of == d].sum()), 3) for d in sorted(set(ds_of))},
                              teacher=args.teacher, world=world, w_teacher=args.w_teacher, w_gt=args.w_gt,
                              lora=sum(p.numel() for p in a_params + b_params), head=sum(p.numel() for p in head))), flush=True)
    log = open(args.out / "train_log.jsonl", "a") if rank == 0 else None
    t0, agg = time.time(), {}

    def save(step):
        if rank != 0:
            return
        named = {id(p): n for n, p in dit.named_parameters() if ".domain." in n}
        hn = {id(p): n for n, p in net.projectors.named_parameters()}
        torch.save(dict(domain={n: p.detach().float().cpu() for n, p in dit.named_parameters() if ".domain." in n},
                        ema_domain={named[k]: v.cpu() for k, v in ema.items() if k in named},
                        head={n: p.detach().float().cpu() for n, p in net.projectors.named_parameters()},
                        ema_head={hn[k]: v.cpu() for k, v in ema.items() if k in hn},
                        rank=args.rank, alpha=float(args.rank), blocks=C.EXEC_BLOCKS, targets=C.TARGETS, step=step,
                        base_ckpt=str(args.base_ckpt), base_options=str(args.base_opt)), args.out / f"step{step}.pt")

    for step in range(1, args.steps + 1):
        frac = step / args.steps
        scale = min(1.0, step / max(args.warmup, 1)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g, lr in zip(optim.param_groups, base_lrs):
            g["lr"] = lr * scale
        optim.zero_grad(set_to_none=True)
        if args.teacher_grad == "pcgrad":
            # conflict-free distillation (PCGrad 2001.06782; auxiliary losses by gradient similarity, Du et al. 1812.02224):
            # the GT group (GT terms + anchor) and the teacher group get separate gradients; where the teacher's gradient
            # points against the GT one, its conflicting component is removed, so the teacher can only move the network
            # in directions that do not raise the GT loss (ARCTIC's GT included)
            lg, lt = 0.0, 0.0
            for _ in range(local_windows):
                v = seqs[np.random.choice(len(seqs), p=w)]
                g_, t_, st = window_loss(net, model.mano_models, v, args, device)
                lg, lt = lg + g_ / local_windows, lt + t_ / local_windows
                for k, val in st.items():
                    agg.setdefault(k, []).extend(val)
            def flat_grad(loss, retain):
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    return torch.zeros(sum(p.numel() for p in params), device=device)
                gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, params)])
            gg = flat_grad(lg, retain=True); gt_ = flat_grad(lt, retain=False)
            if world > 1:
                dist.all_reduce(gg); dist.all_reduce(gt_); gg /= world; gt_ /= world
            dot = float(gg @ gt_); n2 = float(gg @ gg)
            agg.setdefault("pc_cos", []).append(dot / max(1e-12, (n2 * float(gt_ @ gt_)) ** 0.5))   # before projection
            agg.setdefault("pc_projected", []).append(float(dot < 0))
            if dot < 0 and n2 > 0:
                gt_ = gt_ - (dot / n2) * gg
            g = gg + gt_; o = 0
            for p in params:
                p.grad = g[o:o + p.numel()].view_as(p).to(p.dtype); o += p.numel()
        else:
            for _ in range(local_windows):
                v = seqs[np.random.choice(len(seqs), p=w)]
                g_, t_, st = window_loss(net, model.mano_models, v, args, device)
                ((g_ + t_) / local_windows).backward()
                for k, val in st.items():
                    agg.setdefault(k, []).extend(val)
            if world > 1:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad); p.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if torch.isfinite(gn):
            optim.step()
            dec = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for p in params:
                    ema[id(p)].mul_(dec).add_(p.detach().float(), alpha=1 - dec)
        if step % args.log_every == 0 and rank == 0:
            rec = dict(step=step, lr=optim.param_groups[0]["lr"], sec=round(time.time() - t0, 1), gnorm=float(gn))
            for k, vals in agg.items():
                rec[k] = round(float(np.median(vals) if k.endswith("_mm") else np.mean(vals)), 4)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); agg = {}
        if step % args.save_every == 0 or step == args.steps:
            save(step)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
