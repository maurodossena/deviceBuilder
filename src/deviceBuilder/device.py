"""The `Device` class: a Hamiltonian, an overlap and the lattice they live on.

`Device` only holds data. The loaders build and return one, and everything else
that acts on a device is a function that takes it as its first argument and
edits it in place, e.g.

    dev = load_from_cp2k("cp2k.out", "SCF")
    upscale(dev, 2, dir=0)

This module also holds the helpers that keep the derived attributes of a device
(`start_orb_per_at`, `max_coupling`) and the set of its blocks consistent.
"""

import numpy as np
from scipy import sparse as sp

EPS_HERMITIAN = 1e-10


class Device:
    """
    A device, i.e. a set of sparse Hamiltonian and overlap blocks together with
    the lattice they are defined on.
    """

    def __init__(self):
        """
        Initialize an empty Device.
        """

        self.H = {}  # Dictionary with the Hamiltonian (hopping) matrices
        self.S = {}  # Dictionary with the Overlap matrices
        self.lattice = None  # Dictionary with the lattice information (L, at, coords)

        self.orb_map = None  # Dictionary with number of orbitals per atom type (expressed as string)
        self.start_orb_per_at = (
            None  # List with the starting orbital index for each atom
        )

        self.Fermi = None  # Fermi level of the system

        self.potential = None  # Potential profile over the device

        self.max_coupling = [0, 0, 0]  # List to store the maximum coupling in each direction

        self.PBC = [False, False, False]  # List to store the periodic boundary conditions in each direction


def refresh_orb_per_at(dev: Device):
    """
    Recompute the starting orbital index of each atom from the current
    atom list. Must be called after anything that adds, removes or
    reorders atoms.
    """
    dev.start_orb_per_at = np.cumsum(
        [0] + [dev.orb_map[at] for at in dev.lattice["at"]]
    )


def update_max_coupling(dev: Device):
    """
    Update the maximum coupling in each direction for the Hamiltonian and Overlap matrices.
    Stores a list of three integers representing the maximum coupling in the x, y, and z directions.
    """
    max_coupling = [0, 0, 0]

    for key in dev.H.keys():
        for i in range(3):
            max_coupling[i] = max(max_coupling[i], abs(key[i]))

    for key in dev.S.keys():
        for i in range(3):
            max_coupling[i] = max(max_coupling[i], abs(key[i]))

    print("The coupling range is: ", max_coupling)

    dev.max_coupling = max_coupling


def add_missing_couplings(M: dict, max_coupling: list, size: int):
    """
    Tail with empty matrices to obtain a "square" set of couplings.

    Parameters:
    -----------
    M : dict
        Hamiltonian or Overlap dictionary, updated in place.
    max_coupling : list
        Coupling range in each direction.
    size : int
        Size of the matrices to be added.
    """
    for i in range(-max_coupling[0], max_coupling[0] + 1):
        for j in range(-max_coupling[1], max_coupling[1] + 1):
            for k in range(-max_coupling[2], max_coupling[2] + 1):
                key = (i, j, k)
                if key not in M:
                    M[key] = sp.csr_matrix((size, size))


def check_if_hermitian(M: dict, name: str):
    """
    Check that a Hamiltonian or Overlap dictionary is Hermitian.

    Parameters:
    -----------
    M : dict
        Hamiltonian or Overlap dictionary.
    name : str
        Name of the matrix ("H" or "S"), used in the error message.
    """

    for key in M.keys():
        key_opposite = (-key[0], -key[1], -key[2])
        if key_opposite not in M:
            raise Exception(
                "The "
                + name
                + " matrices are not Hermitian. Missing the opposite coupling for key: "
                + str(key)
            )
        if M[key].shape != M[key_opposite].shape:
            raise Exception(
                "The "
                + name
                + " matrices are not Hermitian. The shape of the opposite coupling for key: "
                + str(key)
                + " is different."
            )
        if (np.abs(M[key] - M[key_opposite].transpose().conj())>EPS_HERMITIAN).nnz != 0:
            raise Exception(
                "The "
                + name
                + " matrices are not Hermitian. The opposite coupling for key: "
                + str(key)
                + " is not the conjugate transpose."
            )


def finalize_couplings(dev: Device, size: int = None):
    """
    Bring the blocks of a device back to their canonical state after couplings
    were added or removed: refresh `max_coupling`, pad the keys to a full box
    and check that `H` and `S` are Hermitian.

    Parameters:
    -----------
    size : int
        Size of the empty blocks to pad with. Defaults to the size of the
        in-cell Hamiltonian block.
    """
    update_max_coupling(dev)

    size_H = dev.H[(0, 0, 0)].shape[0] if size is None else size
    size_S = dev.S[(0, 0, 0)].shape[0] if size is None else size
    add_missing_couplings(dev.H, dev.max_coupling, size_H)
    add_missing_couplings(dev.S, dev.max_coupling, size_S)

    check_if_hermitian(dev.H, "H")
    check_if_hermitian(dev.S, "S")
