"""Ion trajectories inside the ideal quadrupole (dimensionless Mathieu units)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    xi: np.ndarray  # dimensionless time, xi = Omega t / 2
    x: np.ndarray  # x / r0
    y: np.ndarray  # y / r0
    hit: bool  # True if the ion reached the rods (r >= r0)
    hit_index: int | None


def integrate_ion(
    a: float,
    q: float,
    x0: float = 0.1,
    y0: float = 0.1,
    vx0: float = 0.0,
    vy0: float = 0.0,
    xi_max: float = 40.0,
    dxi: float = 0.005,
    xi_start: float = 0.0,
    r_wall: float = 1.0,
) -> Trajectory:
    """Integrate x'' + (a - 2q cos 2xi) x = 0 and y'' - (a - 2q cos 2xi) y = 0 (RK4).

    Positions are in units of r0. Integration stops when the ion reaches
    ``r_wall`` (the rods); that is a circular approximation of the electrode
    surface, good enough for teaching. Real instruments: ions need hundreds
    of RF cycles to traverse the filter, here ``xi_max=40`` is ~13 cycles.
    """
    n = int(round(xi_max / dxi))
    xi_arr = xi_start + dxi * np.arange(n + 1)
    xs = np.empty(n + 1)
    ys = np.empty(n + 1)
    state = np.array([x0, vx0, y0, vy0], float)

    def f(xi, s):
        g = a - 2 * q * np.cos(2 * xi)
        return np.array([s[1], -g * s[0], s[3], g * s[2]])

    xs[0], ys[0] = x0, y0
    hit_index = None
    last = n
    for i in range(n):
        xi = xi_arr[i]
        k1 = f(xi, state)
        k2 = f(xi + dxi / 2, state + dxi / 2 * k1)
        k3 = f(xi + dxi / 2, state + dxi / 2 * k2)
        k4 = f(xi + dxi, state + dxi * k3)
        state = state + dxi / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[i + 1], ys[i + 1] = state[0], state[2]
        if np.hypot(state[0], state[2]) >= r_wall:
            hit_index = i + 1
            last = i + 1
            break
    sl = slice(0, last + 1)
    return Trajectory(xi_arr[sl], xs[sl], ys[sl], hit_index is not None, hit_index)
