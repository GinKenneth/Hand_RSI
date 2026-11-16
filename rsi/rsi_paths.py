"""Every machine-specific location of the pipeline, in one place. Each one can be overridden by an environment variable.

  (root)           this directory: the code, and the working outputs the rounds write next to it
                   (captures/, preds/, models/, rounds/R<k>/, loopA_runs/, sealed/, bucket/)
  ACE_EVAL_ROOT    the ace_eval package (default: ../ace_eval)
  RSI_PYTHON       python of the environment with torch, smplx and the ACE-Ego-Hand dependencies (default: this one)
  ACE_OFFICIAL     clone of github.com/ggxxii/ACE-Ego-Hand, commit 59da83b, with its third_party/ and ckpt/ set up as in
                   its README (default: capture/ace_official)
  ACE_CKPT         released ACE-Ego-Hand K-given checkpoint, ace_ego_hand_k.pt (default: checkpoints/ace_ego_hand_k.pt)
  ACE_EVAL_MANO_DIR  converted MANO models (see ACE-Ego-Hand's scripts/convert_mano_pkls.py), read by ace_eval
  WILOR_MINI_ROOT  directory holding wilor_mini/ and its pretrained weights (image-evidence harness modules)
  RSI_DATA_RAW     raw dataset downloads: HOT3D/, ARCTIC/, HOI4D/ (data/convert_*.py)
  RSI_ARCHIVE      where bulky per-segment arrays of finished rounds are moved (rounds/archive_arrays.py)
  RSI_TMPDIR       temporary files of the rounds
  RSI_SANDBOX_HOME, RSI_SANDBOX_UID, RSI_CLAUDE_BIN, RSI_CODEX_ROOT   Loop A / Loop C agent sandbox (loopA/sandbox.py)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_env = os.environ.get
ROOT = Path(__file__).resolve().parent
ACE_EVAL = Path(_env('ACE_EVAL_ROOT') or ROOT.parent / 'ace_eval')
PY = _env('RSI_PYTHON') or sys.executable
TORCHRUN = _env('RSI_TORCHRUN') or str(Path(PY).parent / 'torchrun')
VENV = Path(PY).parent.parent
ACE_OFFICIAL = Path(_env('ACE_OFFICIAL') or ROOT / 'capture' / 'ace_official')
ACE_CKPT = _env('ACE_CKPT') or str(ROOT / 'checkpoints' / 'ace_ego_hand_k.pt')
WILOR_ROOT = _env('WILOR_MINI_ROOT') or str(ROOT / 'third_party' / 'wilor_root')
DATA_RAW = Path(_env('RSI_DATA_RAW') or ROOT / 'datasets')
ARCHIVE = Path(_env('RSI_ARCHIVE') or ROOT / 'archive')
TMP = _env('RSI_TMPDIR') or _env('TMPDIR') or '/tmp'
SANDBOX_HOME = _env('RSI_SANDBOX_HOME') or str(Path.home())
SANDBOX_UID = int(_env('RSI_SANDBOX_UID') or 23001)
CLAUDE_BIN = _env('RSI_CLAUDE_BIN') or '/opt/rsi/claude'
CODEX_ROOT = Path(_env('RSI_CODEX_ROOT') or ROOT / 'third_party' / 'codex')
STATUS_LOG = Path(_env('RSI_STATUS_LOG') or ROOT / 'logs' / 'rounds.md')

if str(ACE_EVAL) not in sys.path:
    sys.path.insert(0, str(ACE_EVAL))
