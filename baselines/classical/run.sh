#!/bin/bash
# Classical post-processing controls (main paper Tab. 2, supplement Tab. S7), CPU only except the WiLoR ensemble.
#   CAPTURES=<dir with <net>/<view> captures> VIEWS=<dir with views> OUT=<output dir> bash run.sh [smoothers|ensemble]
# Captures: theta2 = the released θ2 weights, theta0 = ACE-Ego-Hand K-given, both from rsi/capture/capture.sh on
# hot3d_val, arctic_val, hot3d_test, arctic_test, hoi4d_test. The ensemble needs WiLoR (WILOR_MINI_ROOT) and one GPU.
# 1. every config on θ2's val captures, scored against θ2 with the empty harness (the gate's Δ, --min-gain 0.003);
# 2. the HOT3D-val-best member of each family runs once on the sealed tests (θ2, and θ0 for the smoothers).
set -eu
export HANDRSI_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}; B=$(cd "$(dirname "$0")" && pwd)
: "$CAPTURES" "$VIEWS" "$OUT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p $OUT/preds $OUT/scores $OUT/logs
$PY $B/make_trees.py $OUT
replay() {  # cfg net view
  local O=$OUT/preds/$2/$3/$1; [ -f $O/DONE ] && return
  mkdir -p $O; ( cd $HANDRSI_ROOT/rsi && $PY -m harness.run --tree $OUT/trees/$1 --cache $CAPTURES/$2/$3 --view $VIEWS/$3 \
    --out $O --jobs ${JOBS:-12} > $OUT/logs/rep_$2_$3_$1.log 2>&1 ) && touch $O/DONE
}
score() {  # cfg net view [ref]
  local O=$OUT/scores/$2_$3_$1.json; [ -f $O ] && return
  ( cd $HANDRSI_ROOT/rsi && $PY -m eval.score --gt $VIEWS/$3/gt --pred $OUT/preds/$2/$3/$1 \
      ${4:+--ref $OUT/preds/$2/$3/$4 --boot 1000 --min-gain 0.003} --out $O > $OUT/logs/score_$2_$3_$1.log 2>&1 )
}
configs() { $PY -c "import json,sys; print(' '.join(json.load(open('$B/configs.json'))[sys.argv[1]]))" $1; }
best() { $PY - $OUT/scores "$@" <<'PY'
import json, sys
d, cs = sys.argv[1], sys.argv[2:]
print(max(cs, key=lambda c: json.load(open(f"{d}/theta2_hot3d_val_{c}.json"))["compare"]["S"]))
PY
}
for v in hot3d_val arctic_val; do replay empty theta2 $v; done
if [ "${1:-smoothers}" = smoothers ]; then
  for fam in smoothers align2d; do
    for c in $(configs $fam); do for v in hot3d_val arctic_val; do replay $c theta2 $v; score $c theta2 $v empty; done; done
  done
  SEL="$(best $(configs smoothers)) $(best $(configs align2d))"; echo "val-selected: $SEL"
  for net in theta2 theta0; do for v in hot3d_test arctic_test hoi4d_test; do
    for c in empty $SEL; do replay $c $net $v; score $c $net $v; done; done; done
else
  for c in $(configs ensemble); do CL_WILOR_CACHE=$OUT/wilor_cache/theta2/hot3d_val replay $c theta2 hot3d_val; score $c theta2 hot3d_val empty; done
  SEL=$(best $(configs ensemble)); echo "val-selected: $SEL"
  for v in hot3d_test arctic_test; do
    CL_WILOR_CACHE=$OUT/wilor_cache/theta2/$v replay $SEL theta2 $v; replay empty theta2 $v; score $SEL theta2 $v; score empty theta2 $v
  done
fi
