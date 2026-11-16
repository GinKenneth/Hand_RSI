"""Classical baseline: Savitzky-Golay smoothing (window, polynomial order)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _filt import apply_all, savgol


def apply(state, seg, params):
    w, p = int(params.get("window", 9)), int(params.get("order", 2))
    return apply_all(state, seg, lambda x, kind: savgol(x, w, p))
