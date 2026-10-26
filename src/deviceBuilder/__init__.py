"""
deviceBuilder package

Build and manipulate sparse Hamiltonian/overlap blocks for transport workflows.
A `Device` only holds data. The loaders build and return one; every other
operation is a function that takes the device as its first argument and edits
it in place:

    from deviceBuilder import load_from_cp2k, upscale, export_data_QUATREX

    dev = load_from_cp2k("cp2k.out", path="./run")
    upscale(dev, 2, dir=0)
    export_data_QUATREX(dev, output_dir="device")
"""

from . import io
from .bands import compute_band_structure
from .constants import HARTREE_TO_EV
from .device import Device, refresh_orb_per_at, update_max_coupling
from .export import export_data_OMEN, export_data_QUATREX
from .glue import glue_other_device
from .loaders import load_from_cp2k, load_from_MALOQ_GAMMA, load_from_wannier90
from .transform import (
    align_with_axis,
    center_cell,
    generate_potential_barrier,
    reduce_coupling_dir,
    remove_atoms,
    shift_energies,
    sort_atoms,
    translate_cell,
    wrap_cell,
)
from .upscale import upscale, upscale_cont
from .visualize import print_isosurface_self_interaction, visualize_lattice

__all__ = [
    "Device",
    "io",
    "HARTREE_TO_EV",
    # Keeping a hand-built device consistent
    "refresh_orb_per_at",
    "update_max_coupling",
    # Loading
    "load_from_cp2k",
    "load_from_MALOQ_GAMMA",
    "load_from_wannier90",
    # Reshaping the lattice
    "align_with_axis",
    "center_cell",
    "generate_potential_barrier",
    "reduce_coupling_dir",
    "remove_atoms",
    "shift_energies",
    "sort_atoms",
    "translate_cell",
    "wrap_cell",
    # Assembling a device
    "glue_other_device",
    "upscale",
    "upscale_cont",
    # Bands and views
    "compute_band_structure",
    "print_isosurface_self_interaction",
    "visualize_lattice",
    # Exporting
    "export_data_OMEN",
    "export_data_QUATREX",
]

__version__ = "0.1.0"
