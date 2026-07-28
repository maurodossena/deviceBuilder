"""
deviceBuilder package

Build and manipulate sparse Hamiltonian/overlap blocks for transport workflows:

    from deviceBuilder import Device

    dev = Device()
    dev.load_from_cp2k_GAMMA("cp2k.out", path="./run")
    dev.export_data_QUATREX(output_dir="device")
"""

from . import io
from .constants import HARTREE_TO_EV
from .device import Device

__all__ = ["Device", "io", "HARTREE_TO_EV"]

__version__ = "0.1.0"
