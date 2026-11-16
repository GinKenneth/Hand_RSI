"""Loop B data prep (model independent, once per training sequence): VAE latents of the whole video at its native
protocol size (HOT3D 480x480, ARCTIC 672x480) and the per-frame 3D GT assembled from the 81-frame GT segments.

  python -m loopB.prep --view data/views/hot3d_train --gpu 0 [--shard 0/4]
-> loopB/cache/<sid>/{latents.pt, gt3d.npz, meta.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopB import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    view = a.view.resolve()
    i, k = map(int, a.shard.split("/"))
    seqs = sorted((view / "segments").glob("*.json"))[i::k]
    sys.path.insert(0, str(C.S / "capture"))
    C.ace_imports()
    from ace_ego_hand.video_vae import encode, load_vae
    from run_ace import read_frames, to_pixels
    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    vae = load_vae(device)
    for sj in seqs:
        d = json.loads(sj.read_text()); sid = sj.stem
        dst = C.CACHE / sid; dst.mkdir(parents=True, exist_ok=True)
        if (dst / "latents.pt").is_file() and (dst / "gt3d.npz").is_file():
            print(sid, "cached", flush=True); continue
        n, W, H = int(d["frames"]), int(d["W"]), int(d["H"])
        K = np.asarray(d["K"], float)
        frames = read_frames(str(view / "videos" / Path(d["video"]).name), n, W, H)
        with torch.no_grad():
            lat = encode(vae, to_pixels(frames), device).float()
        lat = lat[0] if lat.dim() == 5 else lat                       # (C, f_lat, h, w)
        torch.save(lat.to(torch.float16).cpu(), dst / "latents.pt")
        J = np.full((n, 2, 21, 3), np.nan, np.float32); tr = np.full((n, 2, 3), np.nan, np.float32)
        rot = np.full((n, 2, 3, 3), np.nan, np.float32); valid = np.zeros((n, 2), bool)
        for sg in d["segments"]:
            g = np.load(view / "gt" / f"{sg['name']}.npz"); sl = slice(sg["start"], sg["end"])
            J[sl], tr[sl], rot[sl], valid[sl] = g["joints_cam"], g["transl"], g["rot"], g["valid"]
        np.savez_compressed(dst / "gt3d.npz", joints_cam=J, transl=tr, rot=rot, valid=valid)
        meta = dict(sequence_id=sid, frames=n, W=W, H=H, K=K.tolist(), latent_shape=list(lat.shape),
                    intr=dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), image_width=W, image_height=H),
                    segments=d["segments"])
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        print(sid, "encoded", list(lat.shape), int(valid.sum()), "GT hand-frames", flush=True)
        del frames, lat
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
