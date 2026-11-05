"""Illustrative ion library for the triple-quadrupole scan-mode demos.

Precursor m/z values are the nominal [M+H]+ of three chlorotriazine herbicides.
Fragment m/z values follow the commonly reported product ions of these
compounds, but **abundances, collision energies and branching are illustrative
teaching numbers, not measured data**. Replace them with values optimised on
the actual 3200 QTRAP before using absolute numbers in a lecture.

``interferent`` is an invented isobaric matrix ion at m/z 216 with different
fragments; it shows why MS/MS is more selective than a full scan.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FragmentChannel:
    mz: float
    ce_opt: float   # eV, collision energy where this channel peaks
    width: float    # eV, width of its energy dependence
    weight: float   # relative weight of the channel


@dataclass(frozen=True)
class Species:
    key: str
    name: str
    mz: float
    abundance: float
    fragments: tuple
    kind: str = "analyte"   # analyte | interferent | background
    e50: float = 12.0       # eV, half-dissociation collision energy
    w50: float = 4.0        # eV, width of the dissociation onset


def _bg(mz, ab):
    return Species(f"bg{mz}", f"background {mz}", mz, ab,
                   (FragmentChannel(mz - 45, 25, 8, 1.0),), kind="background")


SPECIES = (
    Species("atrazine", "atrazine", 216, 1.0, (
        FragmentChannel(174, 20, 7, 1.0),
        FragmentChannel(132, 30, 8, 0.8),
        FragmentChannel(104, 38, 8, 0.7),
        FragmentChannel(96, 40, 8, 0.5),
    )),
    Species("simazine", "simazine", 202, 0.6, (
        FragmentChannel(132, 22, 7, 1.0),
        FragmentChannel(124, 26, 7, 0.5),
        FragmentChannel(104, 36, 8, 0.6),
    )),
    Species("terbuthylazine", "terbuthylazine", 230, 0.8, (
        FragmentChannel(174, 18, 7, 1.0),
        FragmentChannel(132, 28, 8, 0.7),
        FragmentChannel(104, 36, 8, 0.5),
    )),
    Species("interferent", "matrix interferent", 216, 0.7, (
        FragmentChannel(156, 22, 7, 1.0),
        FragmentChannel(128, 32, 8, 0.6),
    ), kind="interferent"),
    _bg(163, 0.15), _bg(181, 0.12), _bg(193, 0.20), _bg(224, 0.14), _bg(247, 0.10),
)


def by_key(key: str) -> Species:
    for s in SPECIES:
        if s.key == key:
            return s
    raise KeyError(key)
