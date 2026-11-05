"""Mathieu stability theory for the linear quadrupole mass filter.

Ideal quadrupole potential (rods at +-r0 from the axis, z along the axis)::

    Phi(x, y, t) = (U + V cos(Omega t)) * (x^2 - y^2) / r0^2

Substituting xi = Omega t / 2 gives the Mathieu equations::

    d^2u/dxi^2 + (a_u - 2 q_u cos 2 xi) u = 0

    a_x = -a_y = 8 z e U / (m r0^2 Omega^2)
    q_x = -q_y = 4 z e V / (m r0^2 Omega^2)

(The overall sign of q only shifts the RF phase and does not change
stability.) Both a and q are proportional to 1/(m/z): at fixed U, V, f, r0
every ion sits on the same straight line through the origin, a = (2U/V) q,
at a position set by its m/z.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import mathieu_a, mathieu_b

E_CHARGE = 1.602176634e-19  # C
AMU = 1.66053906660e-27  # kg


def boundary_lower_a0(q):
    """Characteristic value a_0(q): lower edge of the x-stability band (a<0 here)."""
    return np.array([mathieu_a(0, qi) for qi in np.atleast_1d(q)])


def boundary_upper_b1(q):
    """Characteristic value b_1(q): upper edge of the x-stability band."""
    return np.array([mathieu_b(1, qi) for qi in np.atleast_1d(q)])


def _tip():
    f = lambda q: -mathieu_a(0, q) - mathieu_b(1, q)
    q = brentq(f, 0.5, 0.9)
    return q, mathieu_b(1, q)


Q_TIP, A_TIP = _tip()  # (0.7060, 0.2370)


def stability_boundary(q):
    """Upper boundary a(q) of the first (positive-a) stability region.

    It is min(-a_0(q), b_1(q)): the y-motion limits the region for q < Q_TIP,
    the x-motion for q > Q_TIP. Valid for 0 <= q <= 0.908 (where it reaches
    a = 0 again).
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    return np.minimum(-boundary_lower_a0(q), boundary_upper_b1(q))


def _monodromy_trace(a, q, n_steps: int = 600):
    """Trace of the Floquet monodromy matrix of u'' + (a - 2q cos 2xi) u = 0.

    The coefficient has period pi in xi. The motion is bounded iff |trace| < 2.
    Vectorised over arbitrary-shaped arrays a, q (RK4, fixed step).
    """
    a, q = np.broadcast_arrays(np.asarray(a, float), np.asarray(q, float))
    h = np.pi / n_steps

    def acc(xi, u):
        return -(a - 2 * q * np.cos(2 * xi)) * u

    def run(u, v):
        xi = 0.0
        for _ in range(n_steps):
            k1u, k1v = v, acc(xi, u)
            k2u, k2v = v + 0.5 * h * k1v, acc(xi + 0.5 * h, u + 0.5 * h * k1u)
            k3u, k3v = v + 0.5 * h * k2v, acc(xi + 0.5 * h, u + 0.5 * h * k2u)
            k4u, k4v = v + h * k3v, acc(xi + h, u + h * k3u)
            u = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            xi += h
        return u, v

    u1, _ = run(np.ones_like(a), np.zeros_like(a))  # u(0)=1, u'(0)=0
    _, v2 = run(np.zeros_like(a), np.ones_like(a))  # u(0)=0, u'(0)=1
    return u1 + v2


def is_stable_x(a, q):
    return np.abs(_monodromy_trace(a, q)) < 2.0


def is_stable_y(a, q):
    return np.abs(_monodromy_trace(-np.asarray(a, float), q)) < 2.0


def is_stable(a, q):
    """True where the ion is stable in BOTH x and y (i.e. transmitted)."""
    return is_stable_x(a, q) & is_stable_y(a, q)


def mathieu_parameters(mz, U, V, f, r0, z: int = 1):
    """(a, q) for an ion of given m/z [Da per charge].

    mz : m/z in Th (Da/charge); U, V in volts (V is zero-to-peak RF amplitude);
    f : RF frequency in Hz; r0 : field radius in metres.
    The charge number cancels with m/z, so ``z`` is only kept for clarity.
    """
    omega = 2 * np.pi * f
    ze_over_m = E_CHARGE / (np.asarray(mz, float) * AMU)  # (z e / m) with m/z = mz
    a = 8 * ze_over_m * U / (r0**2 * omega**2)
    q = 4 * ze_over_m * V / (r0**2 * omega**2)
    return a, q


def scan_line_slope(U, V):
    """Slope a/q of the mass-scan line: 2U/V."""
    return 2.0 * U / V


def uv_ratio_for_slope(slope):
    """U/V ratio that gives a scan line of the given slope a/q."""
    return slope / 2.0


def tip_mz(V, f, r0):
    """m/z whose (a, q) lands exactly on the stability-region tip for a given V.

    q = 4 e V / (m r0^2 Omega^2) = Q_TIP  ->  m/z = 4 e V / (Q_TIP r0^2 Omega^2).
    """
    omega = 2 * np.pi * f
    return 4 * E_CHARGE * V / (Q_TIP * r0**2 * omega**2 * AMU)
