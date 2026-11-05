"""Spectra produced by the three triple-quadrupole scan modes (unit-resolution model)."""
from __future__ import annotations

import numpy as np

from .fragmentation import branching

PEAK_SIGMA = 0.55  # Da, shape of a displayed peak


def gaussian(x, mu, sigma=PEAK_SIGMA):
    return np.exp(-0.5 * ((np.asarray(x, float) - mu) / sigma) ** 2)


def full_scan_spectrum(species, mz_grid):
    """Q1 scan, Q2/Q3 RF-only: every ion appears at its precursor m/z."""
    y = np.zeros_like(np.asarray(mz_grid, float))
    for s in species:
        y = y + s.abundance * gaussian(mz_grid, s.mz)
    return y


def product_ion_spectrum(sp, ce, mz_grid):
    """Q1 fixed on sp.mz, Q2 at collision energy ``ce``, Q3 scanned: product ions of one species."""
    y = np.zeros_like(np.asarray(mz_grid, float))
    for mz, frac in branching(sp, ce).items():
        y = y + sp.abundance * frac * gaussian(mz_grid, mz)
    return y


def mrm_intensity(sp, q1, q3, ce, hw=1.2):
    """Relative signal of one species for the transition q1 -> q3 at collision energy ce."""
    if abs(sp.mz - q1) > hw:
        return 0.0
    return sp.abundance * sum(f for mz, f in branching(sp, ce).items() if abs(mz - q3) <= hw)
