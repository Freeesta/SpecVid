import numpy as np
import pytest

from core.quad import (
    A_TIP, Q_TIP, integrate_ion, is_stable, mathieu_parameters,
    stability_boundary, tip_mz,
)


def test_tip_coordinates():
    assert Q_TIP == pytest.approx(0.706, abs=1e-3)
    assert A_TIP == pytest.approx(0.237, abs=1e-3)


def test_floquet_matches_analytic_boundary():
    q = np.array([0.2, 0.4, 0.6, 0.8])
    ab = stability_boundary(q)
    assert is_stable(ab - 0.004, q).all()
    assert (~is_stable(ab + 0.004, q)).all()


def test_origin_neighbourhood_and_far_outside():
    assert is_stable(0.02, 0.3)
    assert not is_stable(0.05, 0.3)  # just above the boundary (~0.045)
    assert not is_stable(0.30, 0.70)
    assert not is_stable(-0.05, 0.3)  # below a_0(q) -> unstable in x


def test_parameters_scale_inverse_with_mass_and_a_over_q():
    a1, q1 = mathieu_parameters(100, 100.0, 600.0, 1e6, 4e-3)
    a2, q2 = mathieu_parameters(200, 100.0, 600.0, 1e6, 4e-3)
    assert q1 == pytest.approx(2 * q2)
    assert a1 / q1 == pytest.approx(2 * 100.0 / 600.0)


def test_tip_mz_round_trip():
    f, r0, V = 1.0e6, 4e-3, 500.0
    m = tip_mz(V, f, r0)
    _, q = mathieu_parameters(m, 0.0, V, f, r0)
    assert q == pytest.approx(Q_TIP, rel=1e-6)


def test_stable_ion_stays_unstable_ion_hits():
    ok = integrate_ion(0.05, 0.40, xi_max=60)
    assert not ok.hit and np.hypot(ok.x, ok.y).max() < 0.5
    bad = integrate_ion(0.20, 0.85, xi_max=60)
    assert bad.hit
