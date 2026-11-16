"""WiLoR's image-space 2D keypoints on given recordings -> the reference that Loop B's teacher pseudo-labels are checked
against (loopB/targets.py quality_weight).

The keypoints come from WiLoR's crop camera mapped back to the full image (ace_eval.baselines.diag_wilor2d.crop_2d,
used unchanged), so they do not depend on any depth or focal assumption. WiLoR's 3D joints projected through K are NOT
used: on HOT3D (fx = 208, wide field of view) they sit ~23 px median from the GT, mostly a wrist offset, while ACE sits
~4 px (checked on 4 GT train recordings; the paper's WiLoR EPE2D on HOT3D is 73 px for the same reason).
Output per segment: <segment>.npz {kp2d: (81, 2, 21, 2) px, NaN where WiLoR has no box for that hand}.

  WILOR_MINI_ROOT=... PYTHONPATH=../ace_eval $PY data/wilor_ref2d.py --data data/views/hot3d_train \
      --out data/hot3d_nogt_wilor2d --seqs h3__P0004_1984bf99,... --device cuda:0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ace_eval.baselines.diag_wilor2d import crop_2d
from ace_eval.baselines.run_wilor import load_pipe, read_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seqs", required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipe(a.device)
    for seq in a.seqs.split(","):
        meta = json.loads((a.data / "segments" / f"{seq}.json").read_text())
        todo = [sg for sg in meta["segments"] if not (a.out / f"{sg['name']}.npz").exists()]
        if not todo:
            continue
        t0 = time.time()
        frames = read_frames(str(a.data / "videos" / f"{seq}.mp4"), None); n = len(frames)
        kp = crop_2d(pipe, frames)
        del frames
        kp2d = np.full((n, 2, 21, 2), np.nan, np.float32)
        for (f, s), v in kp.items():
            kp2d[f, s] = v
        k = 0
        for sg in todo:
            if sg["end"] <= n:
                np.savez_compressed(a.out / f"{sg['name']}.npz", kp2d=kp2d[sg["start"]:sg["end"]]); k += 1
        print(f"{seq}: {n} frames, {len(kp)} hand boxes, {k} segments, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
