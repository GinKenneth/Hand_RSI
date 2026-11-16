"""Classical baseline: One-Euro filter (Casiez et al. 2012), causal; signals normalised by _filt.SCALE."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _filt import apply_all, one_euro, SCALE


def apply(state, seg, params):
    mc, b = float(params.get("mincutoff", 1.0)), float(params.get("beta", 0.0))
    return apply_all(state, seg, lambda x, kind: one_euro(x, mc, b, scale=SCALE[kind]))
