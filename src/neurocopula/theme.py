"""
A single matplotlib theme, so every figure the library produces reads as one
system.

The palette is not chosen by taste. It is a validated categorical set in
which adjacent hues stay distinguishable under the common forms of colour
vision deficiency, used in FIXED SLOT ORDER and never cycled. Scatter plots
compare all pairs of series at once rather than just neighbours, which is a
stricter requirement, so they are capped at the first three slots -- the
subset that clears the all-pairs threshold. That cap is why every
model-comparison figure here shows at most three series (typically observed
data, neural copula, vine copula) and folds anything further into separate
panels.

Sequential magnitude uses one blue hue light-to-dark; polarity (a
correlation matrix, an error matrix) uses a blue/red diverging pair with a
NEUTRAL GRAY midpoint, so "no dependence" reads as absence rather than as a
third colour.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "CATEGORICAL", "SERIES", "SEQUENTIAL_CMAP", "DIVERGING_CMAP",
    "INK", "INK_MUTED", "SURFACE", "GRID", "apply_theme", "series_color",
]

#: Categorical hues in fixed slot order. Never reorder, never cycle.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

#: Semantic slots for the recurring three-way comparison. Colour follows the
#: ENTITY, so "observed data" is the same blue in every figure in the library
#: regardless of which models appear beside it.
SERIES = {
    "data": CATEGORICAL[0],
    "neurocopula": CATEGORICAL[1],
    "vine": CATEGORICAL[2],
    "theory": "#52514e",
}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2de"

#: One hue, light to dark, for continuous magnitude.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "nc_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

#: Blue <-> red with a neutral gray midpoint, for signed quantities.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nc_diverging",
    ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a62726"],
)


def series_color(name: str, fallback_slot: int = 0) -> str:
    """Colour for a named entity, falling back to a categorical slot."""
    return SERIES.get(name, CATEGORICAL[fallback_slot % len(CATEGORICAL)])


def apply_theme() -> None:
    """
    Apply the library's matplotlib rcParams: recessive grid and axes, ink-
    coloured text, thin marks, no top/right spines.

    Called automatically by every plotting function, so figures look
    consistent whether or not you invoke it yourself. Call it directly at
    the top of a notebook if you want your own plots to match.
    """
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9,
        "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })
