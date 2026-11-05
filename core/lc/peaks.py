"""Chromatographic peak shapes (minimal for now; EMG, van Deemter etc. come later)."""
import numpy as np


def gaussian_peak(t, t_r, sigma, height=1.0):
    """Gaussian elution profile centred on retention time ``t_r`` (same unit as t)."""
    return height * np.exp(-0.5 * ((np.asarray(t, float) - t_r) / sigma) ** 2)
