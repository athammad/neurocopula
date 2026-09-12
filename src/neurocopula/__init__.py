"""
neurocopula -- neural copulas built on normalizing flows.

A copula separates a joint distribution into its marginals and its
dependence structure. Classical copula models pick a parametric family for
the dependence -- Gaussian, Student-t, Clayton -- and inherit whatever tail
behaviour that family happens to impose. `neurocopula` instead learns the
dependence structure with a normalizing flow (`zuko`'s Neural Spline Flow),
so the shape is inferred from the data rather than assumed, while the
marginals are still reproduced exactly by their empirical distribution
functions.

Quick start::

    from neurocopula import NeuroCopula, datasets

    df = datasets.make_clayton(n=4000, theta=2.0)
    model = NeuroCopula(copula_mode=True).fit(df)

    model.tail_dependence("x1", "x2", q=0.02, tail="lower")
    model.joint_exceedance_probability({"x1": ("<", -2.0), "x2": ("<", -2.0)})
    model.conditional_probability("x2", "<", -2.0, given={"x1": -3.0})
    model.sample(10_000)

The comparison baseline, `VineCopula`, needs the optional `pyvinecopulib`
dependency and is imported lazily -- ``from neurocopula import VineCopula``
raises a clear install message if it is missing.

For gridded data, `SpatialProbabilisticLayer` combines a per-location copula
with a spatial dependence model, so that a region selected *after* fitting can
be queried jointly::

    from neurocopula import SpatialProbabilisticLayer

    layer = SpatialProbabilisticLayer(["rain", "temp", "wind"]).fit(data, coords)
    cells = layer.select(bbox=(-2, 2, 103, 107))
    layer.exceedance_probability({"rain": (">", 50)}, cells, mode="any")
"""

from __future__ import annotations

import importlib

from . import datasets, marginals, metrics, plotting, semiparametric, spatial, theme
from .core import NeuroCopula
from .marginals import (
    EmpiricalCopulaTransform,
    StandardizeTransform,
    pseudo_observations,
)
from .semiparametric import GroupedMarginalTransform, SemiParametricMarginal
from .spatial import SpatialDependence

__version__ = "0.1.0"

__all__ = [
    "NeuroCopula",
    "SpatialProbabilisticLayer",
    "SeasonalLayers",
    "SpatialDependence",
    "RegionSet",
    "SemiParametricMarginal",
    "GroupedMarginalTransform",
    "VineCopula",
    "EmpiricalCopulaTransform",
    "StandardizeTransform",
    "pseudo_observations",
    "datasets",
    "marginals",
    "metrics",
    "plotting",
    "semiparametric",
    "spatial",
    "theme",
    "benchmark",
    "layer",
    "seasonal",
    "regions",
    "io",
    "__version__",
]


def __getattr__(name: str):
    """
    Lazily expose the optional pieces, so importing `neurocopula` never
    requires `pyvinecopulib` and never pays for `benchmark`'s imports unless
    they are actually used.

    `importlib.import_module` is used rather than a plain ``from . import x``:
    the latter re-enters this very function through the import machinery's
    fromlist handling and recurses forever.
    """
    # (attribute name) -> (submodule, whether to return the module itself)
    lazy = {
        "VineCopula": (".vine", False),
        "benchmark": (".benchmark", True),
        # layer/io pull in torch-heavy and xarray-dependent code; keep them
        # off the critical path of a plain `import neurocopula`.
        "SpatialProbabilisticLayer": (".layer", False),
        "SeasonalLayers": (".seasonal", False),
        "RegionSet": (".regions", False),
        "regions": (".regions", True),
        "seasonal": (".seasonal", True),
        "layer": (".layer", True),
        "io": (".io", True),
    }
    if name in lazy:
        modname, want_module = lazy[name]
        module = importlib.import_module(modname, __name__)
        value = module if want_module else getattr(module, name)
        globals()[name] = value  # cache, so this runs at most once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
