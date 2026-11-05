# SpecVid

Physics models and [Manim](https://www.manim.community/) animations for teaching
LC-UV → ESI → triple quadrupole / QTRAP (environmental chemistry MSc course).
Language of the videos: English. Equations are shown on screen.

## Layout

```
core/        physics only (numpy/scipy), no graphics, unit-tested
  quad/      Mathieu stability, scan line, ion trajectories
scenes/      one Manim scene per teaching concept (style.py = shared look)
tests/       pytest
```

Planned modules: `core/esi`, `core/lc`, `core/uv`, `core/msms` (see roadmap in chat/notes).

## Setup

System packages (Debian/Ubuntu): `libpango1.0-dev libcairo2-dev texlive-latex-extra texlive-fonts-recommended texlive-science dvisvgm cm-super ffmpeg`

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Render

```bash
manim -pql scenes/quad_stability.py    QuadStability     # draft 480p15
manim -qh  scenes/quad_trajectories.py QuadTrajectories  # final 1080p60 for YouTube
```

Output goes to `media/` (git-ignored).

## Conventions

- Mathieu form: `u'' + (a_u - 2 q_u cos 2ξ) u = 0`, `ξ = Ωt/2`,
  `a = 8zeU/(m r0² Ω²)`, `q = 4zeV/(m r0² Ω²)`. The sign of q only shifts the RF phase.
- Trajectories use a circular approximation of the rod surface (`r = r0`) and ~13 RF
  cycles; real instruments need hundreds of cycles to traverse the filter.
- Instrument parameters (RF frequency, r0, voltage ranges) for the 3200 QTRAP are
  **not** hard-coded: take them from the Sciex manual before showing absolute m/z values.
