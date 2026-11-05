"""Triple-quadrupole scan modes: full scan, product ion scan, MRM.

Render (draft):
  manim -ql scenes/qqq_scans.py FullScan
  manim -ql scenes/qqq_scans.py ProductIonScan
  manim -ql scenes/qqq_scans.py MRM
"""
import numpy as np
from manim import *

from core.lc import gaussian_peak
from core.msms import (
    PEAK_SIGMA, SPECIES, branching, by_key, full_scan_spectrum, optimal_ce, product_ion_spectrum,
)
from scenes.qqq_common import *
from scenes.style import ACCENT, BAD, BG, FG, MUTED, OK, WARM

ABUND = np.array([s.abundance for s in SPECIES])


def _static_weights(t):
    return ABUND


class FullScan(Scene):
    """Q1 scans, Q2 and Q3 are RF-only: a spectrum of the precursor ions."""

    def construct(self):
        self.camera.background_color = BG
        clock = make_clock(self)
        ramp = Ramp(9.0, 36.0, 150.0, 260.0)

        title = Text("Full scan (Q1 scan)", font_size=34, color=FG).to_edge(UP, buff=0.25)
        inst = build_instrument()
        self.add(disclaimer())
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(inst), run_time=1.5)
        st = VGroup(state_text("Q1", "scans m/z"), state_text("Q2", "RF only, no CID"), state_text("Q3", "RF only"))
        self.play(FadeIn(st))

        q1 = lambda t: ramp.value(t) if ramp.active(t) else None
        stream = IonStream(clock, _static_weights, q1, lambda t: None, lambda t: None)
        self.add(stream)
        leg = legend()
        self.play(FadeIn(leg))
        cap = caption("ESI delivers a mixture of ions; Q1 will transmit one m/z at a time")
        self.play(FadeIn(cap))
        wait_until(self, clock, 9.0)

        # spectrum builds while Q1 scans 150 -> 260
        mz = np.arange(150, 260.01, 0.25)
        ax, labels = spectrum_axes([150, 260, 20], [0, 2.0, 1], y_len=2.7, center=(0, -1.35))
        total = full_scan_spectrum(SPECIES, mz)
        self.play(Create(ax), FadeIn(labels))
        curve = reveal_curves(ax, [(mz, total, FG, 3)], ramp.value, clock)
        readout = make_readout("Q1 =", lambda: float(ramp.value(clock.get_value())), x=-3.9)
        self.add(curve, readout)
        self.play(FadeOut(cap))
        cap2 = caption("Signal is recorded as a function of the m/z transmitted by Q1", y=-3.65)
        self.play(FadeIn(cap2))
        wait_until(self, clock, 36.0 + LAG + 0.5)
        self.play(FadeOut(readout), FadeOut(cap2))

        # annotate
        def peak_label(mz0, txt, color, dy=0.15):
            return Text(txt, font_size=18, color=color).next_to(ax.c2p(mz0, full_scan_spectrum(SPECIES, np.array([mz0]))[0]), UP, buff=dy)
        l1 = peak_label(202, "simazine", WARM)
        l3 = peak_label(230, "terbuthylazine", OK)
        l2 = VGroup(Text("atrazine", font_size=18, color=ACCENT), Text("+ interferent", font_size=18, color=BAD)).arrange(DOWN, buff=0.03)
        l2.next_to(ax.c2p(216, 1.7), UP, buff=0.1)
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(l3))
        cap3 = caption("m/z 216 is two different ions, but a full scan sees one peak", color=BAD)
        self.play(FadeIn(cap3))
        self.wait(5)
        cap4 = caption("Full scan: good for discovery, but no structure information and low selectivity", color=FG, font_size=24)
        self.play(FadeOut(cap3))
        self.play(FadeIn(cap4))
        self.wait(5)


class ProductIonScan(Scene):
    """Q1 fixed on the precursor, CID in Q2, Q3 scans: the fragmentation fingerprint."""

    def construct(self):
        self.camera.background_color = BG
        clock = make_clock(self)
        T_ON = 6.0
        CE = 25.0
        ramp = Ramp(10.0, 34.0, 80.0, 230.0)
        T_CE0, T_CE1 = 44.0, 58.0
        ce_ramp = Ramp(T_CE0, T_CE1, 5.0, 50.0)

        def q1(t):
            return 216.0 if t >= T_ON else None

        def ce_of_tag(t):
            return CE if t < T_CE0 else float(ce_ramp.value(t))

        def ce_fn(t):
            if t < T_ON:
                return None
            return ce_of_tag(t)

        def q3(t):
            return float(ramp.value(t)) if ramp.active(t) else None

        title = Text("Product ion scan", font_size=34, color=FG).to_edge(UP, buff=0.25)
        inst = build_instrument()
        self.add(disclaimer())
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(inst), run_time=1.5)
        stream = IonStream(clock, _static_weights, q1, ce_fn, q3)
        self.add(stream)
        leg = legend()
        self.play(FadeIn(leg))
        st1 = VGroup(state_text("Q1", "fixed: m/z 216", ACCENT), state_text("Q2", f"CID, CE = {CE:g} eV", WARM), state_text("Q3", "scans m/z", FG))
        self.play(FadeIn(st1))
        cap = caption("Q1 selects the precursor, Q2 fragments it, Q3 scans the products")
        self.play(FadeIn(cap))
        wait_until(self, clock, 10.0)

        mz = np.arange(80, 230.01, 0.25)
        ax, labels = spectrum_axes([80, 230, 20], [0, 0.8, 0.2], y_len=2.7, center=(0, -1.35))
        atr, itf = by_key("atrazine"), by_key("interferent")
        y_atr = product_ion_spectrum(atr, CE, mz)
        y_itf = product_ion_spectrum(itf, CE, mz)
        self.play(Create(ax), FadeIn(labels))
        curves = reveal_curves(ax, [(mz, y_atr, ACCENT, 3), (mz, y_itf, BAD, 3)], ramp.value, clock)
        readout = make_readout("Q3 =", lambda: float(ramp.value(clock.get_value())), x=2.8)
        self.add(curves, readout)
        self.play(FadeOut(cap))
        cap2 = caption("Only ions selected by Q1 (m/z 216) appear: the fingerprint of that precursor")
        self.play(FadeIn(cap2))
        wait_until(self, clock, 34.0 + LAG + 0.5)
        self.play(FadeOut(readout))

        fam = [(174, "a"), (132, "a"), (104, "a"), (96, "a"), (156, "i"), (128, "i"), (216, "p")]
        tags = VGroup()
        for m0, f in fam:
            g = np.array([m0], float)
            ya, yi = product_ion_spectrum(atr, CE, g)[0], product_ion_spectrum(itf, CE, g)[0]
            yv, c = {"a": (ya, ACCENT), "i": (yi, BAD), "p": (max(ya, yi), FG)}[f]
            tags.add(Text(f"{m0}", font_size=18, color=c).next_to(ax.c2p(m0, yv), UP, buff=0.08))
        self.play(FadeIn(tags))
        self.play(FadeOut(cap2))
        cap3 = caption("Two families of fragments: the 216 peak of a full scan was two different molecules", color=BAD, font_size=24)
        self.play(FadeIn(cap3))
        self.wait(5)

        # ---- Part B: collision-energy ramp ----
        self.play(FadeOut(cap3), FadeOut(curves), FadeOut(tags), FadeOut(st1[2]), FadeOut(st1[1]))
        st2 = VGroup(state_text("Q2", "CID, CE ramp", WARM), state_text("Q3", "RF only", FG))
        self.play(FadeIn(st2))
        wait_until(self, clock, T_CE0 - 3.0)
        cap4 = caption("Raise the collision energy and acquire again: the fragment pattern changes", font_size=24)
        self.play(FadeIn(cap4))
        mz_c = np.arange(80, 230.01, 0.5)

        def dyn():
            ce = ce_of_tag(clock.get_value() - LAG)
            ys = [(product_ion_spectrum(atr, ce, mz_c), ACCENT), (product_ion_spectrum(itf, ce, mz_c), BAD)]
            g = VGroup()
            for y, c in ys:
                pts = [ax.c2p(x, yy) for x, yy in zip(mz_c, y)]
                g.add(VMobject(stroke_color=c, stroke_width=3).set_points_as_corners(pts))
            return g

        ce_read = make_readout("CE =", lambda: ce_of_tag(clock.get_value() - LAG), x=3.3, y=0.45, decimals=0, unit="eV", num_color=WARM)
        dyn_curves = always_redraw(dyn)
        self.add(dyn_curves, ce_read)
        wait_until(self, clock, T_CE1 + LAG + 1.0)
        self.play(FadeOut(dyn_curves), FadeOut(ce_read), FadeOut(cap4), FadeOut(stream), FadeOut(st2), FadeOut(st1[0]), FadeOut(ax), FadeOut(labels), FadeOut(inst), FadeOut(leg))

        # ---- Part C: breakdown curves ----
        ces = np.linspace(0, 60, 241)
        ax2 = Axes(x_range=[0, 60, 10], y_range=[0, 1.0, 0.2], x_length=10.0, y_length=3.6,
                   axis_config={"color": MUTED, "include_tip": False, "font_size": 20},
                   x_axis_config={"numbers_to_include": [0, 10, 20, 30, 40, 50, 60]},
                   y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0], "decimal_number_config": {"num_decimal_places": 1}}
                   ).move_to([0, -0.55, 0])
        xl = Text("collision energy (eV)", font_size=20, color=FG).next_to(ax2, DOWN, buff=0.15)
        yl = Text("fraction of precursor", font_size=18, color=MUTED).rotate(PI / 2).next_to(ax2, LEFT, buff=0.2)
        chan = [(216, "216 (survivor)", MUTED), (174, "174", ACCENT), (132, "132", OK), (104, "104", WARM), (96, "96", BAD)]
        self.play(Create(ax2), FadeIn(xl), FadeIn(yl))
        lines, keyrow = VGroup(), VGroup()
        for m0, name, c in chan:
            ys = np.array([branching(atr, ce).get(m0, 0.0) for ce in ces])
            lines.add(ax2.plot_line_graph(ces, ys, line_color=c, add_vertex_dots=False, stroke_width=3))
            keyrow.add(VGroup(Line(ORIGIN, RIGHT * 0.3, color=c), Text(name, font_size=18, color=FG)).arrange(RIGHT, buff=0.1))
        keyrow.arrange(RIGHT, buff=0.4).move_to([0, 2.45, 0])
        self.play(Create(lines), FadeIn(keyrow), run_time=3)
        ce_best = optimal_ce(atr, 174)
        vline = DashedLine(ax2.c2p(ce_best, 0), ax2.c2p(ce_best, 1.0), color=ACCENT, stroke_width=2)
        tag = Text(f"216 → 174 is maximal near {ce_best:g} eV", font_size=22, color=ACCENT).next_to(ax2.c2p(ce_best, 1.0), UP, buff=0.12).shift(RIGHT * 2.0)
        self.play(Create(vline), FadeIn(tag))
        cap5 = caption("Each fragment has its own best CE: this is how MRM transitions are optimised", font_size=24)
        self.play(FadeIn(cap5))
        self.wait(5)


class MRM(Scene):
    """Q1 and Q3 both fixed: two filters in series, cycled over several transitions."""

    TRANS = [("simazine", 202, 132), ("atrazine", 216, 174), ("terbuthylazine", 230, 174)]
    RT = {"simazine": 3.2, "atrazine": 4.2, "terbuthylazine": 5.6}  # min, illustrative
    DWELL = 0.5

    def construct(self):
        self.camera.background_color = BG
        clock = make_clock(self)
        T_ON, T_B = 5.0, 24.0
        ces = {k: optimal_ce(by_key(k), f) for k, _, f in self.TRANS}
        LC0, LC_RATE = 2.0, 0.2  # min, min per video second in part B
        sig = 0.12

        def lc_time(t):
            return LC0 + (t - T_B) * LC_RATE

        def idx_at(t):
            return int((t - T_B) / self.DWELL) % 3

        def tr_at(t):
            if t < T_ON:
                return None
            if t < T_B:
                return ("atrazine", 216, 174)
            return self.TRANS[idx_at(t)]

        q1 = lambda t: None if tr_at(t) is None else float(tr_at(t)[1])
        q3 = lambda t: None if tr_at(t) is None else float(tr_at(t)[2])
        ce_fn = lambda t: None if tr_at(t) is None else ces[tr_at(t)[0]]

        def weights(t):
            if t < T_B:
                return ABUND
            lc = lc_time(t)
            w = ABUND.copy()
            for i, sp in enumerate(SPECIES):
                if sp.key in self.RT:
                    w[i] = sp.abundance * (0.03 + gaussian_peak(lc, self.RT[sp.key], sig))
            return w

        title = Text("Multiple reaction monitoring (MRM)", font_size=34, color=FG).to_edge(UP, buff=0.25)
        inst = build_instrument()
        self.add(disclaimer())
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(inst), run_time=1.5)
        stream = IonStream(clock, weights, q1, ce_fn, q3)
        self.add(stream)
        leg = legend()
        self.play(FadeIn(leg))
        st = VGroup(state_text("Q1", "fixed: m/z 216", ACCENT), state_text("Q2", f"CID, {ces['atrazine']:g} eV", WARM), state_text("Q3", "fixed: m/z 174", OK))
        self.play(FadeIn(st))
        wait_until(self, clock, T_ON)

        # ---- Part A: one transition, two filters ----
        rows = VGroup(
            Text("1.  Q1 = 216 passes atrazine AND the isobaric interferent; everything else is blocked", font_size=21, color=FG),
            Text("2.  Q2 fragments them: atrazine gives 174, 132, 104, 96; the interferent gives 156, 128", font_size=21, color=FG),
            Text("3.  Q3 = 174 transmits only atrazine's 174 fragment: the interferent is rejected", font_size=21, color=OK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.9, 0])
        trans = MathTex(r"216 \rightarrow 174", font_size=56, color=ACCENT).move_to([0, -2.7, 0])
        for r in rows:
            self.play(FadeIn(r, shift=UP * 0.15), run_time=1.0)
            self.wait(3.2)
        self.play(Write(trans))
        wait_until(self, clock, T_B - 3.0)
        self.play(FadeOut(rows), FadeOut(trans))

        # ---- Part B: several transitions + chromatogram ----
        cap = caption("LC-MS/MS: the instrument cycles through the transitions while compounds elute", font_size=24)
        self.play(FadeIn(cap))
        table_rows = VGroup()
        colors = {"simazine": WARM, "atrazine": ACCENT, "terbuthylazine": OK}
        for name, a, b in self.TRANS:
            table_rows.add(Text(f"{a} → {b}   {name}   CE {ces[name]:g} eV", font_size=18, color=colors[name]))
        table_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-3.7, -1.2, 0])
        hdr = Text("MRM method", font_size=21, color=FG).next_to(table_rows, UP, buff=0.3).align_to(table_rows, LEFT)
        hl = Rectangle(width=table_rows.width + 0.3, height=0.42, stroke_color=FG, stroke_width=2).move_to(table_rows[0])
        hl.add_updater(lambda m: m.move_to(table_rows[idx_at(clock.get_value())]) if clock.get_value() >= T_B else None)
        self.play(FadeIn(hdr), FadeIn(table_rows), FadeIn(hl))

        ax = Axes(x_range=[2, 7, 1], y_range=[0, 1.1, 0.5], x_length=6.0, y_length=2.6,
                  axis_config={"color": MUTED, "include_tip": False, "font_size": 18},
                  x_axis_config={"numbers_to_include": [2, 3, 4, 5, 6, 7]},
                  y_axis_config={"numbers_to_include": []}).move_to([3.1, -1.25, 0])
        xl = Text("retention time (min)", font_size=18, color=FG).next_to(ax, DOWN, buff=0.12)
        yl = Text("MRM signal", font_size=16, color=MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.15)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        tt = np.linspace(2, 7, 400)
        series = []
        for name, _, _ in self.TRANS:
            sp = by_key(name)
            series.append((tt, sp.abundance * gaussian_peak(tt, self.RT[name], sig), colors[name], 3))
        lcx = lambda t: lc_time(t + 0.0)
        curves = reveal_curves(ax, series, lcx, clock)
        self.add(curves)
        wait_until(self, clock, T_B + 25.0 + LAG + 0.5)
        self.play(FadeOut(cap))
        cap2 = caption("Two mass filters in series: selective, sensitive, quantitative", color=OK, font_size=26)
        self.play(FadeIn(cap2))
        self.wait(5)
