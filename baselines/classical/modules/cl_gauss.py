"""Classical baseline: zero-phase Gaussian smoothing (sigma in frames)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _filt import apply_all, gauss


def apply(state, seg, params):
    s = float(params.get("sigma", 1.0))
    return apply_all(state, seg, lambda x, kind: gauss(x, s))
