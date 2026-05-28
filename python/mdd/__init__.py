"""Multi-Dimensional Deconvolution for acoustic Green's function extraction.

Companion package to the Julia `AcousticDisguising` module. Reads reverberant
pressure + normal-velocity data produced by `scripts/greens/reverb.jl`,
extracts Green's functions via pylops-based MDD, and writes them in the
HDF5 schema that `scripts/hologram/synthesize.jl` consumes via `load_gf_mdd`
on the Julia side.
"""

from .mdd import MDDParams, mdd_extract
from .io import load_reverb, save_gfs
from .wavefield_separation import (
    SHTSeparationParams,
    decompose_outer_pin_sht,
)
from .wavefield_separation_local import (
    LocalPWSeparationParams,
    decompose_outer_pin_local_pw,
    decompose_outer_pin_local_pvn,
)

__all__ = [
    "MDDParams", "mdd_extract",
    "load_reverb", "save_gfs",
    "SHTSeparationParams", "decompose_outer_pin_sht",
    "LocalPWSeparationParams",
    "decompose_outer_pin_local_pw",
    "decompose_outer_pin_local_pvn",
]
__version__ = "1.0.0"
