import numpy as np
import pytest

from core.lc import gaussian_peak
from core.msms import (
    SPECIES, branching, by_key, fragment_outcome, full_scan_spectrum,
    mrm_intensity, optimal_ce, product_ion_spectrum,
)


@pytest.mark.parametrize("ce", [0, 10, 25, 50])
def test_branching_sums_to_one(ce):
    for s in SPECIES:
        assert sum(branching(s, ce).values()) == pytest.approx(1.0)


def test_survival_falls_with_ce_and_high_energy_channels_take_over():
    atr = by_key("atrazine")
    assert branching(atr, 5)[216] > 0.8 > branching(atr, 40)[216]
    assert branching(atr, 15)[174] > branching(atr, 15)[104]
    assert branching(atr, 45)[104] > branching(atr, 45)[174]


def test_optimal_ce_is_reasonable():
    atr = by_key("atrazine")
    assert 12 <= optimal_ce(atr, 174) <= 28


def test_fragment_outcome_matches_branching():
    atr = by_key("atrazine")
    rng = np.random.default_rng(0)
    out = [fragment_outcome(atr, 25, u) for u in rng.random(20000)]
    frac174 = out.count(174) / len(out)
    assert frac174 == pytest.approx(branching(atr, 25)[174], abs=0.02)


def test_full_scan_cannot_separate_isobars():
    grid = np.array([216.0])
    total = full_scan_spectrum(SPECIES, grid)[0]
    assert total == pytest.approx(by_key("atrazine").abundance + by_key("interferent").abundance, rel=1e-6)


def test_product_ion_spectrum_peaks_at_fragments():
    grid = np.arange(80, 231, 1.0)
    y = product_ion_spectrum(by_key("atrazine"), 25, grid)
    assert grid[np.argmax(y)] == 174


def test_mrm_selectivity():
    atr, intf, sim = by_key("atrazine"), by_key("interferent"), by_key("simazine")
    assert mrm_intensity(atr, 216, 174, 20) > 0.3
    assert mrm_intensity(intf, 216, 174, 20) == 0.0   # passes Q1, blocked by Q3
    assert mrm_intensity(sim, 216, 174, 20) == 0.0    # blocked by Q1
    assert mrm_intensity(intf, 216, 156, 22) > 0.0


def test_gaussian_peak():
    assert gaussian_peak(4.0, 4.0, 0.1) == 1.0
