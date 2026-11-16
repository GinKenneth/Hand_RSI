#!/bin/bash
# capture.sh NODE CKPT_DIR_OR_RELEASE VIEW GPU [SHARD] : raw ungated ACE outputs of NODE on a view -> captures/NODE/<view>/
# (official inference, 480/672 native, segment context). Release ACE: CKPT=release. Locations: rsi_paths.py.
NODE=$1; CK=$2; VIEW=$3; GPU=$4; SHARD=${5:-0/1}
S=$(cd "$(dirname "$0")/.." && pwd); V=$S/data/views/$VIEW; OUT=$S/captures/$NODE/$VIEW; mkdir -p $OUT
PY=${RSI_PYTHON:-python}; ACE=${ACE_OFFICIAL:-$S/capture/ace_official}
if [ "$CK" = release ]; then CKPT=${ACE_CKPT:-$S/checkpoints/ace_ego_hand_k.pt}; OPT=$ACE/options/ace_ego_hand_k.yml
else CKPT=$CK/ace_ego_hand_k.pt; OPT=$CK/ace_ego_hand_k.yml; fi
CUDA_VISIBLE_DEVICES=$GPU $PY $S/capture/run_ace.py --data $V --video-dir $V/videos \
  --out $OUT --ckpt $CKPT --opt $OPT --gpu 0 --shard $SHARD
