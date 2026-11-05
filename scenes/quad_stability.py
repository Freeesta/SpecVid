"""Scene 1 - Mathieu stability diagram and the mass scan line.

Render:  manim -pql scenes/quad_stability.py QuadStability
"""
import numpy as np
from manim import *

from core.quad import A_TIP, Q_TIP, is_stable, stability_boundary
from scenes.style import ACCENT, BAD, BG, FG, MUTED, OK, WARM

IONS = [(100, WARM), (150, ACCENT), (200, OK)]  # m/z, colour
SLOPE = 0.325            # a/q = 2U/V, just below the tip slope A_TIP/Q_TIP
MT_RANGE = (60.0, 260.0)  # m/z of the ion sitting at the tip, swept by ramping U and V


class QuadStability(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---- precompute transmission of each ion along the scan (vectorised) ----
        mt_grid = np.linspace(*MT_RANGE, 801)
        trans = {}
        for mz, _ in IONS:
            q = Q_TIP * mt_grid / mz
            trans[mz] = is_stable(SLOPE * q, q) & (q < 1.0)

        # ---- title and equations ----
        title = Text("Quadrupole mass filter: stability", font_size=36, color=FG).to_edge(UP, buff=0.3)
        phi = MathTex(r"\Phi=(U+V\cos\Omega t)\,\frac{x^{2}-y^{2}}{r_0^{2}}", font_size=40, color=FG)
        mat = MathTex(r"\frac{d^{2}u}{d\xi^{2}}+\bigl(a_u-2q_u\cos 2\xi\bigr)u=0,\quad \xi=\tfrac{\Omega t}{2}",
                      font_size=40, color=FG)
        aq = MathTex(r"a=\frac{8zeU}{m\,r_0^{2}\Omega^{2}},\qquad q=\frac{4zeV}{m\,r_0^{2}\Omega^{2}}",
                     font_size=40, color=FG)
        eqs = VGroup(phi, mat, aq).arrange(DOWN, buff=0.45)
        self.play(Write(title))
        for e in eqs:
            self.play(Write(e), run_time=1.6)
        note = Text("a and q are both proportional to 1 / (m/z)", font_size=28, color=ACCENT).next_to(eqs, DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(2)

        # ---- move equations to the right column, build the diagram ----
        right_eqs = VGroup(mat.copy(), aq.copy()).arrange(DOWN, buff=0.25).scale(0.62)
        right_eqs.move_to(RIGHT * 3.6 + UP * 2.6)
        self.play(FadeOut(phi), FadeOut(note), Transform(mat, right_eqs[0]), Transform(aq, right_eqs[1]))

        axes = Axes(
            x_range=[0, 1.0, 0.2], y_range=[0, 0.3, 0.1],
            x_length=5.9, y_length=4.7,
            axis_config={"color": MUTED, "include_tip": False, "font_size": 24, "decimal_number_config": {"num_decimal_places": 1}},
            x_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]},
            y_axis_config={"numbers_to_include": [0.1, 0.2, 0.3]},
        ).move_to(LEFT * 3.5 + DOWN * 0.5)
        xl = MathTex("q", font_size=34, color=FG).next_to(axes.x_axis, RIGHT, buff=0.1).shift(DOWN * 0.1)
        yl = MathTex("a", font_size=34, color=FG).next_to(axes.y_axis, UP, buff=0.1)
        self.play(Create(axes), Write(xl), Write(yl))

        q_edge = np.linspace(0, 0.9080, 200)
        upper = stability_boundary(q_edge)
        pts = [axes.c2p(q, a) for q, a in zip(q_edge, upper)] + [axes.c2p(q_edge[-1], 0), axes.c2p(0, 0)]
        region = Polygon(*pts, stroke_color=ACCENT, stroke_width=3, fill_color=ACCENT, fill_opacity=0.25)
        tip = Dot(axes.c2p(Q_TIP, A_TIP), color=FG, radius=0.06)
        tip_lbl = MathTex(rf"({Q_TIP:.3f},\,{A_TIP:.3f})", font_size=26, color=FG).next_to(tip, RIGHT, buff=0.12)
        region_lbl = Text("stable in x and y", font_size=22, color=ACCENT).move_to(axes.c2p(0.38, 0.045))
        self.play(FadeIn(region), run_time=1.5)
        self.play(FadeIn(tip), Write(tip_lbl), FadeIn(region_lbl))
        self.wait(1)

        # ---- scan line ----
        line = Line(axes.c2p(0, 0), axes.c2p(0.92, SLOPE * 0.92), color=FG, stroke_width=3)
        line_lbl = MathTex(r"\frac{a}{q}=\frac{2U}{V}", font_size=30, color=FG).next_to(line.get_end(), UP, buff=0.1).shift(LEFT * 0.6)
        self.play(Create(line), Write(line_lbl))
        self.wait(0.5)

        # ---- ions on the line, driven by the scan tracker ----
        mt = ValueTracker(MT_RANGE[0])

        def idx():
            return int(np.clip(np.searchsorted(mt_grid, mt.get_value()), 0, len(mt_grid) - 1))

        def ion_dot(mz, colour):
            def build():
                q = Q_TIP * mt.get_value() / mz
                if q > 0.98:
                    return Dot(axes.c2p(0, 0), radius=0.0001, fill_opacity=0)
                inside = trans[mz][idx()]
                d = Dot(axes.c2p(q, SLOPE * q), radius=0.11 if inside else 0.08, color=colour)
                d.set_fill(colour, opacity=1.0 if inside else 0.45)
                if inside:
                    d.set_stroke(FG, width=2)
                return d
            return always_redraw(build)

        dots = VGroup(*[ion_dot(mz, c) for mz, c in IONS])
        legend = VGroup(*[
            VGroup(Dot(color=c, radius=0.08), Text(f"m/z {mz}", font_size=22, color=c)).arrange(RIGHT, buff=0.15)
            for mz, c in IONS
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(axes.c2p(0.17, 0.255))
        self.play(FadeIn(dots), FadeIn(legend))

        # ---- mass spectrum builds up as U and V are ramped ----
        spec_axes = Axes(
            x_range=[MT_RANGE[0], MT_RANGE[1], 50], y_range=[0, 1.2, 1],
            x_length=5.6, y_length=1.9,
            axis_config={"color": MUTED, "include_tip": False, "font_size": 20},
            x_axis_config={"numbers_to_include": [100, 150, 200, 250]},
        ).move_to(RIGHT * 3.6 + DOWN * 1.65)
        spec_lbl = Text("detector signal vs m/z (U, V ramped together)", font_size=20, color=MUTED).next_to(spec_axes, DOWN, buff=0.15)
        spec_y = Text("ion current", font_size=18, color=MUTED).rotate(PI / 2).next_to(spec_axes, LEFT, buff=0.15)
        self.play(Create(spec_axes), FadeIn(spec_lbl), FadeIn(spec_y))

        signal = sum(trans[mz].astype(float) for mz, _ in IONS)

        def spectrum():
            i = idx()
            m = VMobject(stroke_color=ACCENT, stroke_width=3)
            pts = [spec_axes.c2p(mt_grid[k], signal[k]) for k in range(0, i + 1, 2)]
            if len(pts) >= 2:
                m.set_points_as_corners(pts)
            return m

        readout = always_redraw(lambda: MathTex(
            rf"m/z_{{\mathrm{{tip}}}}={mt.get_value():.0f}", font_size=30, color=FG
        ).move_to(RIGHT * 3.6 + UP * 0.2))
        self.add(always_redraw(spectrum), readout)
        self.play(mt.animate.set_value(MT_RANGE[1]), run_time=16, rate_func=linear)
        self.wait(1)

        msg = Text("Only ions inside the region are transmitted", font_size=26, color=OK).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(msg))
        self.wait(2)
