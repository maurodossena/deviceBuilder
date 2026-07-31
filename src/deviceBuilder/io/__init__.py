"""
File-format readers and writers.

One module per format:

* `cp2k`      : CP2K output files and the binary CSR matrices they produce.
* `wannier90` : Wannier90 `.wout` and `_hr.dat` files.
* `xyz`       : extended XYZ files.
* `omen`      : the binary matrix format read by OMEN.
"""

from .cp2k import  read_cp2k_file, read_csr
from .omen import print_bin
from .wannier90 import read_hr_dat, read_wannier_wout
from .xyz import read_xyz

__all__ = [
    "print_bin",
    "read_cp2k_file",
    "read_hr_dat",
    "read_wannier_wout",
    "read_xyz",
    "read_csr",
]
