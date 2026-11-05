"""Collision-induced dissociation (CID) in Q2: a deliberately simple model.

* The dissociated fraction rises sigmoidally with collision energy (CE).
* Among dissociated ions, each product channel i has a Gaussian dependence on
  CE centred on ``ce_opt`` with width ``width``; channels are renormalised, so
  low-energy channels dominate at low CE and high-energy channels at high CE.

Teaching model only: no sequential fragmentation, no ion losses.
"""
from __future__ import annotations

import numpy as np


def dissociated_fraction(sp, ce):
    return 1.0 / (1.0 + np.exp(-(np.asarray(ce, float) - sp.e50) / sp.w50))


def branching(sp, ce):
    """{m/z: fraction of the precursor flux}, survival (key sp.mz) included. Sums to 1."""
    d = float(dissociated_fraction(sp, ce))
    w = np.array([f.weight * np.exp(-0.5 * ((ce - f.ce_opt) / f.width) ** 2) for f in sp.fragments])
    total = w.sum()
    out = {sp.mz: 1.0 - d}
    if total <= 0:
        out[sp.mz] = 1.0
        return out
    for f, wi in zip(sp.fragments, w):
        out[f.mz] = out.get(f.mz, 0.0) + d * wi / total
    return out


def fragment_outcome(sp, ce, u):
    """m/z of the ion leaving Q2 for a uniform random number u in [0, 1)."""
    acc = 0.0
    last = sp.mz
    for mz, p in branching(sp, ce).items():
        acc += p
        last = mz
        if u < acc:
            return mz
    return last


def optimal_ce(sp, frag_mz, ce_grid=None):
    """CE (eV) that maximises the branching to the fragment ``frag_mz``."""
    grid = np.arange(2.0, 70.0, 0.5) if ce_grid is None else np.asarray(ce_grid, float)
    vals = [branching(sp, c).get(frag_mz, 0.0) for c in grid]
    return float(grid[int(np.argmax(vals))])
