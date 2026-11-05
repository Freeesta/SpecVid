"""Shared building blocks for the triple-quadrupole scenes.

* ``build_instrument``: schematic ESI -> Q0 -> Q1 -> Q2 -> Q3 -> detector.
* ``IonStream``: a continuous stream of ions whose fate at each stage is a pure
  function of the video clock. Every ion is "tagged" with the instrument
  settings in force when it enters Q1 (as in a real instrument, where flight
  time is microseconds), then filtered by Q1, fragmented in Q2 and filtered by
  Q3. The stream and the spectra come from the same physics (core.msms).
* ``reveal_curves``: draws spectra progressively as a scan advances.
"""
from __future__ import annotations

import numpy as np
from manim import *

from core.msms import SPECIES, branching
from scenes.style import ACCENT, BAD, BG, FG, MUTED, OK, WARM

X0, X1 = -6.4, 6.4          # ion path (scene x)
Y_AXIS = 2.45               # ion path (scene y)
STAGES = {
    "Q0": (-5.8, -4.8),
    "Q1": (-4.4, -1.6),
    "Q2": (-1.2, 1.6),
    "Q3": (2.0, 4.8),
    "DET": (5.3, 6.3),
}
HW = 1.2                    # half-width (Da) of the mass-filter window used for the ion stream
LAP = 3.0                   # seconds an ion needs to cross the whole instrument

COLORS = {
    "atrazine": ACCENT, "simazine": WARM, "terbuthylazine": OK, "interferent": BAD,
}


def species_color(sp):
    return COLORS.get(sp.key, MUTED)


def _s(x):
    return (x - X0) / (X1 - X0)


S_Q1A, S_Q1M = _s(STAGES["Q1"][0]), _s((STAGES["Q1"][0] + STAGES["Q1"][1]) / 2)
S_Q2A = _s(STAGES["Q2"][0])
S_Q3A, S_Q3M = _s(STAGES["Q3"][0]), _s((STAGES["Q3"][0] + STAGES["Q3"][1]) / 2)
S_DET = _s(STAGES["DET"][0] + 0.2)
LAG = (S_DET - S_Q1A) * LAP  # seconds from Q1 entry to detection


class Ramp:
    """Linear ramp of a mass-analyser setting between two clock times."""

    def __init__(self, t0, t1, a, b):
        self.t0, self.t1, self.a, self.b = t0, t1, a, b

    def active(self, t):
        return self.t0 <= t <= self.t1

    def value(self, t):
        f = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return self.a + f * (self.b - self.a)


def build_instrument():
    """Return (group, state_label_factory)."""
    g = VGroup()
    g.add(Line([X0, Y_AXIS, 0], [X1, Y_AXIS, 0], color=MUTED, stroke_width=1, stroke_opacity=0.5))
    names = {"Q0": "Q0", "Q1": "Q1", "Q2": "Q2 (collision cell)", "Q3": "Q3", "DET": "Det."}
    for key, (xa, xb) in STAGES.items():
        box = Rectangle(width=xb - xa, height=0.9, stroke_color=MUTED, stroke_width=2,
                        fill_color="#161b22", fill_opacity=1).move_to([(xa + xb) / 2, Y_AXIS, 0])
        lab = Text(names[key], font_size=20, color=FG).move_to(box)
        g.add(box, lab)
    esi = VGroup(Dot([X0, Y_AXIS, 0], radius=0.09, color=FG),
                 Text("ESI", font_size=18, color=FG).move_to([X0 + 0.05, Y_AXIS + 0.5, 0]))
    g.add(esi)
    return g


def state_text(stage, text, color=FG, font_size=19):
    xa, xb = STAGES[stage]
    return Text(text, font_size=font_size, color=color).move_to([(xa + xb) / 2, Y_AXIS - 0.72, 0])


def legend(keys=None, y=1.35):
    items = []
    for sp in SPECIES:
        if sp.kind == "background":
            continue
        if keys is not None and sp.key not in keys:
            continue
        items.append(VGroup(Dot(radius=0.07, color=species_color(sp)),
                            Text(f"{sp.name} {sp.mz:g}", font_size=17, color=FG)).arrange(RIGHT, buff=0.1))
    items.append(VGroup(Dot(radius=0.07, color=MUTED), Text("other ions", font_size=17, color=FG)).arrange(RIGHT, buff=0.1))
    return VGroup(*items).arrange(RIGHT, buff=0.4).move_to([0, y, 0])


_OUTCOME_CACHE = {}


def fragment_outcome(sp, ce, u):
    """Cached version of core.msms.fragment_outcome (CE rounded to 0.5 eV)."""
    key = (sp.key, round(ce * 2))
    table = _OUTCOME_CACHE.get(key)
    if table is None:
        acc, table = 0.0, []
        for mz, p in branching(sp, key[1] / 2).items():
            acc += p
            table.append((acc, mz))
        _OUTCOME_CACHE[key] = table
    for acc, mz in table:
        if u < acc:
            return mz
    return table[-1][1]


class IonStream(VGroup):
    """Stream of ions; all behaviour is a function of ``clock`` (video seconds)."""

    def __init__(self, clock, weights_fn, q1_fn, ce_fn, q3_fn, n_ions=220, seed=3, n_laps=600):
        super().__init__()
        self.clock, self.weights_fn = clock, weights_fn
        self.q1_fn, self.ce_fn, self.q3_fn = q1_fn, ce_fn, q3_fn
        rng = np.random.default_rng(seed)
        self.n = n_ions
        self.phi = (np.arange(n_ions) + rng.uniform(0, 0.9, n_ions)) / n_ions
        self.u_sp = rng.random((n_ions, n_laps))
        self.u_f = rng.random((n_ions, n_laps))
        self.yoff = rng.uniform(-0.3, 0.3, n_ions)
        for _ in range(n_ions):
            self.add(Dot(radius=0.05, stroke_width=0))
        self.add_updater(lambda m: m._refresh())

    def _refresh(self):
        T = self.clock.get_value()
        v = 1.0 / LAP
        for k, dot in enumerate(self.submobjects):
            pos = v * T + self.phi[k]
            lap = int(pos)
            s = pos - lap
            t0 = T - s / v                 # time this ion started its lap
            t1 = t0 + S_Q1A / v            # time it enters Q1 (instrument state tag)
            w = np.asarray(self.weights_fn(t0), float)
            idx = int(np.searchsorted(np.cumsum(w) / w.sum(), self.u_sp[k, lap]))
            sp = SPECIES[min(idx, len(SPECIES) - 1)]
            mz, color = sp.mz, species_color(sp)

            stop = S_DET + 0.03
            c1 = self.q1_fn(t1)
            if c1 is not None and abs(mz - c1) > HW:
                stop = S_Q1M
            else:
                ce = self.ce_fn(t1)
                mz2 = mz if ce is None else fragment_outcome(sp, ce, self.u_f[k, lap])
                if mz2 != mz and s >= S_Q2A:
                    color = interpolate_color(color, WHITE, 0.45)
                c3 = self.q3_fn(t1)
                if c3 is not None and abs(mz2 - c3) > HW:
                    stop = S_Q3M
            x = X0 + s * (X1 - X0)
            dot.move_to([x, Y_AXIS + self.yoff[k], 0])
            dot.set_fill(color, opacity=1.0 if s <= stop else 0.0)


def spectrum_axes(x_range, y_range, y_len=3.0, x_len=11.0, center=(0, -1.15), y_label="relative intensity"):
    step = x_range[2]
    nums = list(np.arange(np.ceil(x_range[0] / step) * step, x_range[1] + 1e-9, step))
    ax = Axes(
        x_range=list(x_range), y_range=list(y_range), x_length=x_len, y_length=y_len,
        axis_config={"color": MUTED, "include_tip": False, "font_size": 20},
        x_axis_config={"numbers_to_include": nums},
    ).move_to([center[0], center[1], 0])
    xl = Text("m/z", font_size=22, color=FG).next_to(ax.x_axis, RIGHT, buff=0.15)
    yl = Text(y_label, font_size=18, color=MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.2)
    return ax, VGroup(xl, yl)


def reveal_curves(axes, series, xfun, clock, lag=LAG):
    """series: [(xs, ys, colour, width)]. Draws each curve up to xfun(clock - lag)."""
    pre = [(np.asarray(xs), np.array([axes.c2p(x, y) for x, y in zip(xs, ys)]), c, w) for xs, ys, c, w in series]

    def build():
        xcur = xfun(clock.get_value() - lag)
        g = VGroup()
        for xs, pts, c, w in pre:
            i = int(np.searchsorted(xs, xcur))
            if i >= 2:
                g.add(VMobject(stroke_color=c, stroke_width=w).set_points_as_corners(pts[:i]))
        return g

    return always_redraw(build)


def make_clock(scene):
    clock = ValueTracker(0.0)
    clock.add_updater(lambda m, dt: m.increment_value(dt))
    scene.add(clock)
    return clock


def caption(text, color=FG, font_size=26, y=-3.65):
    return Text(text, font_size=font_size, color=color).move_to([0, y, 0])


def disclaimer():
    return Text("illustrative model, not measured data", font_size=15, color=MUTED).move_to([0, 3.1, 0])


def make_readout(label, value_fn, x, y=0.85, decimals=1, unit="", num_color=FG):
    """Label + live number (+ unit) that re-aligns itself as the number changes width."""
    lab = Text(label, font_size=20, color=FG).move_to([x, y, 0])
    num = DecimalNumber(0, num_decimal_places=decimals, font_size=24, color=num_color)
    un = Text(unit, font_size=18, color=FG)
    num.add_updater(lambda m: m.set_value(value_fn()).next_to(lab, RIGHT, buff=0.12))
    un.add_updater(lambda m: m.next_to(num, RIGHT, buff=0.1))
    return VGroup(lab, num, un)


def wait_until(scene, clock, t):
    """Wait (video time) until the clock reaches t; no-op if already past."""
    d = t - clock.get_value()
    if d > 0.02:
        scene.wait(d)
