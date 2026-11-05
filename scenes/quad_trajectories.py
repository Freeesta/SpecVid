"""Scene 2 - ion motion in the quadrupole: stable vs unstable (Mathieu a, q).

Render:  manim -pql scenes/quad_trajectories.py QuadTrajectories
"""
import numpy as np
from manim import *

from core.quad import integrate_ion, is_stable
from scenes.style import ACCENT, BAD, BG, FG, MUTED, OK, WARM

R0 = 1.5          # scene units per r0
ROD_R = 0.6       # schematic rod radius (not to scale)
CENTER = LEFT * 3.9 + DOWN * 0.3
XI_MAX = 40.0

CASES = [
    # label, a, q, colour
    ("Ion A: (a, q) inside the stability region", 0.05, 0.40, OK),
    ("Ion B: (a, q) outside the region", 0.20, 0.85, BAD),
]


class QuadTrajectories(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Ion motion between the rods", font_size=36, color=FG).to_edge(UP, buff=0.3)
        eqs = VGroup(
            MathTex(r"\frac{d^{2}x}{d\xi^{2}}+\bigl(a-2q\cos 2\xi\bigr)\,x=0", font_size=36, color=FG),
            MathTex(r"\frac{d^{2}y}{d\xi^{2}}-\bigl(a-2q\cos 2\xi\bigr)\,y=0", font_size=36, color=FG),
        ).arrange(DOWN, buff=0.3).move_to(RIGHT * 3.6 + UP * 2.3)
        self.play(Write(title))
        self.play(Write(eqs), run_time=2)

        # ---- cross-section: r0 circle and four rods ----
        r0_circle = DashedVMobject(Circle(radius=R0, color=MUTED, stroke_width=2), num_dashes=40).move_to(CENTER)
        rods_x = [Circle(radius=ROD_R, fill_opacity=0.9, stroke_width=2).move_to(CENTER + RIGHT * s * (R0 + ROD_R)) for s in (1, -1)]
        rods_y = [Circle(radius=ROD_R, fill_opacity=0.9, stroke_width=2).move_to(CENTER + UP * s * (R0 + ROD_R)) for s in (1, -1)]
        r0_lbl = MathTex("r_0", font_size=26, color=MUTED).move_to(CENTER + RIGHT * R0 * 0.55 + UP * 0.2)
        r0_line = Line(CENTER, CENTER + RIGHT * R0, color=MUTED, stroke_width=2)
        self.play(FadeIn(VGroup(*rods_x, *rods_y)), Create(r0_circle), Create(r0_line), Write(r0_lbl))

        # rods change polarity with the RF: colour follows the sign of (a - 2q cos 2xi)
        xi_t = ValueTracker(0.0)
        params = {"a": 0.05, "q": 0.40}

        def recolor(_=None):
            g = params["a"] - 2 * params["q"] * np.cos(2 * xi_t.get_value())
            cx, cy = (BAD, ACCENT) if g > 0 else (ACCENT, BAD)
            for r in rods_x:
                r.set_fill(cx, opacity=0.9).set_stroke(cx)
            for r in rods_y:
                r.set_fill(cy, opacity=0.9).set_stroke(cy)
        recolor()
        for r in rods_x + rods_y:
            r.add_updater(recolor)
        pol = VGroup(
            Dot(color=BAD, radius=0.07), Text("positive", font_size=20, color=FG),
            Dot(color=ACCENT, radius=0.07), Text("negative rod potential", font_size=20, color=FG),
        ).arrange(RIGHT, buff=0.12).move_to(CENTER + DOWN * (R0 + 2 * ROD_R + 0.45))
        self.play(FadeIn(pol))

        # ---- x(xi), y(xi) plot ----
        axes = Axes(
            x_range=[0, XI_MAX, 10], y_range=[-1.2, 1.2, 1],
            x_length=6.0, y_length=2.8,
            axis_config={"color": MUTED, "include_tip": False, "font_size": 20},
            x_axis_config={"numbers_to_include": [0, 10, 20, 30, 40]},
            y_axis_config={"numbers_to_include": [-1, 0, 1]},
        ).move_to(RIGHT * 3.6 + DOWN * 1.35)
        ax_x = MathTex(r"\xi=\Omega t/2", font_size=26, color=MUTED).next_to(axes, DOWN, buff=0.1)
        wall_hi = DashedLine(axes.c2p(0, 1), axes.c2p(XI_MAX, 1), color=BAD, stroke_width=2)
        wall_lo = DashedLine(axes.c2p(0, -1), axes.c2p(XI_MAX, -1), color=BAD, stroke_width=2)
        self.play(Create(axes), Write(ax_x), Create(wall_hi), Create(wall_lo))
        self.wait(0.5)

        for label, a, q, colour in CASES:
            self.run_case(label, a, q, colour, xi_t, params, axes, rods_x + rods_y)

        self.wait(2)

    def run_case(self, label, a, q, colour, xi_t, params, axes, rods):
        params["a"], params["q"] = a, q
        tr = integrate_ion(a, q, x0=0.10, y0=0.10, xi_max=XI_MAX, dxi=0.005)
        stable = bool(is_stable(a, q))
        assert stable != tr.hit or stable  # sanity: stable ions must not hit
        step = 8
        pts_xy = np.stack([tr.x, tr.y], axis=1)

        def to_scene(p):
            return CENTER + RIGHT * R0 * p[0] + UP * R0 * p[1]

        def idx():
            return int(np.clip(np.searchsorted(tr.xi, xi_t.get_value()), 1, len(tr.xi) - 1))

        caption = Text(label, font_size=22, color=colour).move_to(RIGHT * 3.6 + UP * 0.95)
        values = MathTex(rf"a={a:.2f},\quad q={q:.2f}", font_size=32, color=colour).move_to(RIGHT * 2.3 + UP * 0.4)
        self.play(FadeIn(caption), FadeIn(values))

        xi_t.set_value(0.0)
        path = always_redraw(lambda: VMobject(stroke_color=colour, stroke_width=3, stroke_opacity=0.9).set_points_as_corners(
            [to_scene(p) for p in pts_xy[: idx() + 1 : step]] + [to_scene(pts_xy[idx()])]
        ) if idx() > 1 else VMobject())
        ion = always_redraw(lambda: Dot(to_scene(pts_xy[idx()]), radius=0.1, color=colour).set_stroke(FG, width=2))

        def curve(series, col):
            def build():
                m = VMobject(stroke_color=col, stroke_width=2.5)
                i = idx()
                if i > 1:
                    m.set_points_as_corners([axes.c2p(tr.xi[k], series[k]) for k in range(0, i + 1, step)])
                return m
            return always_redraw(build)

        cx, cy = curve(tr.x, ACCENT), curve(tr.y, WARM)
        leg = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=ACCENT), MathTex(r"x/r_0", font_size=24, color=FG)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=WARM), MathTex(r"y/r_0", font_size=24, color=FG)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.4).next_to(axes, UP, buff=0.05).align_to(axes, RIGHT)
        self.add(path, ion, cx, cy, leg)

        xi_end = float(tr.xi[-1])
        self.play(xi_t.animate.set_value(xi_end), run_time=0.22 * xi_end, rate_func=linear)

        if tr.hit:
            end = to_scene(pts_xy[-1])
            self.play(Flash(end, color=BAD, line_length=0.25, flash_radius=0.4))
            note = Text("hits a rod and is lost", font_size=26, color=BAD).move_to(RIGHT * 3.6 + DOWN * 3.75)
        else:
            note = Text("bounded motion: ion is transmitted", font_size=26, color=OK).move_to(RIGHT * 3.6 + DOWN * 3.75)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(FadeOut(VGroup(path, ion, cx, cy, leg, caption, values, note)))
        self.remove(path, ion, cx, cy)
