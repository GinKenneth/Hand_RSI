"""Regularize poorly observed hand shape toward the MANO population mean.

Shape is a property of the hand, shared throughout a segment. The mean amount
of image evidence controls one constant shrinkage factor for each hand slot.
This avoids trusting a free shape estimate when a hand is mostly out of view.

`power` applies the shrinkage factor that many times in one step: power 4 equals the four identical copies of this
prior that the R3 teacher ran (shape_prior, a2__shape_prior, temporal_shape_prior, a3__shape_prior) in the final
shape (between them no module changed betas; the norm bound only binds on the first application). The modules between
the old copies now see the final shrinkage earlier; dev effect of the replacement alone: HOT3D S -0.003 %, ARCTIC +0.025 %.
"""
import numpy as np


def apply(state, seg, params):
    strength = float(params.get("strength", 1.0))
    visibility = np.asarray(seg.raw["exists_2d"], dtype=np.float64)[:seg.n]
    uncertainty = np.mean(1.0 - np.clip(visibility, 0.0, 1.0), axis=0)
    shrinkage = np.clip(strength * uncertainty, 0.0, 1.0)
    scale = (1.0 - shrinkage) ** float(params.get("power", 1.0))
    radius = float(params.get("radius", 0.0))
    if radius > 0:
        # MANO coefficients have a unit Gaussian prior. A typical shape lies
        # near radius sqrt(number of coefficients); bound unusually large fits.
        norm = np.linalg.norm(np.mean(state["betas"], axis=0), axis=-1)
        scale *= np.minimum(1.0, radius / np.maximum(norm, 1e-8))
    state["betas"] = state["betas"] * scale[None, :, None]
    return seg.mano(state)
