"""FPS of a node (network + harness), for the FPS column of the table.

Timing in the spirit of ACE App. C.4 (one 81-frame ARCTIC clip, median of >= 3 runs after a warm-up), on our hardware
(RTX PRO 5000, recorded in the output):
  network  VAE encode + predict_video (K-given, tiled, tap 15) of one 81-frame segment, cuda-synchronised
  harness  the node's harness chain on that segment's raw outputs (the modules run where they choose: CPU, or GPU for
           image models); the agent-written chain runs in the sandbox like everywhere else (no network)
  fps      81 / (t_network + t_harness)
  python rounds/fps.py --ckpt release|MODEL_DIR --tree TREE --view arctic_test --node NODE --gpu 0 --out fps.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import rsi_paths  # noqa: E402
ACE = rsi_paths.ACE_OFFICIAL
RELEASE = Path(rsi_paths.ACE_CKPT)
PY = rsi_paths.PY

HARNESS_TIMER = r'''
import json, sys, time, numpy as np
from pathlib import Path
from harness.core import Segment, load_chain, run_segment
tree, raw, K, W, H, video, start, n = sys.argv[1:9]
chain = load_chain(Path(tree)); r = dict(np.load(raw)); K = json.loads(K)
ts = []
for i in range(4):
    seg = Segment(r, K, int(W), int(H), video=video, start=int(start), n=int(n))
    t0 = time.perf_counter(); run_segment(chain, seg); ts.append(time.perf_counter() - t0)
print(json.dumps(ts[1:]))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tree', required=True, type=Path)
    ap.add_argument('--view', default='arctic_test'); ap.add_argument('--node', required=True)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    view = S / 'data/views' / a.view
    sj = sorted((view / 'segments').glob('*.json'))[0]
    d = json.loads(sj.read_text()); sg = d['segments'][0]
    n = sg['end'] - sg['start']; K = np.asarray(d['K']); W, H = int(d['W']), int(d['H'])
    video = str(view / 'videos' / Path(d['video']).name)
    raw = S / 'captures' / a.node / a.view / f"{sg['name']}.npz"
    # network
    sys.path[:0] = [str(ACE), str(ACE / 'third_party'), str(S / 'capture')]
    import torch
    import yaml
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    cwd = os.getcwd(); os.chdir(ACE)
    from ace_ego_hand.inference import predict_video
    from ace_ego_hand.models.geodit_model import GeoDitModel
    from ace_ego_hand.video_vae import encode, load_vae
    from run_ace import read_frames, to_pixels
    dev = torch.device('cuda', a.gpu); torch.cuda.set_device(dev)
    ck = RELEASE if a.ckpt == 'release' else Path(a.ckpt) / 'ace_ego_hand_k.pt'
    opt = ACE / 'options' / 'ace_ego_hand_k.yml' if a.ckpt == 'release' else Path(a.ckpt) / 'ace_ego_hand_k.yml'
    model = GeoDitModel(yaml.safe_load(open(opt)), dev); model.load_inference(str(ck))
    vae = load_vae(dev)
    frames = read_frames(video, sg['end'], W, H)[sg['start']:sg['end']]
    intr = {'fx': float(K[0, 0]), 'fy': float(K[1, 1]), 'cx': float(K[0, 2]), 'cy': float(K[1, 2]), 'image_width': W, 'image_height': H}
    ts = []
    for i in range(a.runs + 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            ctrl = encode(vae, to_pixels(frames), dev).float()
            predict_video(model, 15, ctrl, intr, 'tiled', dev, tile_w=22)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    t_net = float(np.median(ts[1:]))
    os.chdir(cwd)
    # harness (sandboxed: agent-written code)
    sys.path.insert(0, str(S / 'loopA'))
    import sandbox
    r = sandbox.run([PY, '-c', HARNESS_TIMER, str(a.tree.resolve()), str(raw), json.dumps(K.tolist()), str(W), str(H), video,
                     str(sg['start']), str(n)], ro=[str(a.tree.resolve()), str(raw), str(view / 'videos')], cwd=str(S),
                    # WiLoR for image-evidence modules, as harness/run.py sets it (the venv is mounted in every sandbox)
                    env=dict(PYTHONPATH=str(S), CUDA_VISIBLE_DEVICES=str(a.gpu),
                             WILOR_MINI_ROOT=rsi_paths.WILOR_ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, net=False, timeout=1800)
    try:
        t_h = float(np.median(json.loads(r.stdout.strip().splitlines()[-1])))
    except (ValueError, IndexError):
        raise SystemExit(f'harness timing failed: {r.stdout[-1500:]}')
    res = dict(segment=sg['name'], frames=n, size=[W, H], gpu=torch.cuda.get_device_name(dev), t_network_s=t_net,
               t_harness_s=t_h, fps_network=n / t_net, fps_total=n / (t_net + t_h), runs=a.runs)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
