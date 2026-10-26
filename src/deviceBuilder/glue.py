"""Joining two devices along a common interface."""

import numpy as np
from scipy import sparse as sp

from .device import Device, refresh_orb_per_at, update_max_coupling
from .lattice import get_orb_from_at


def glue_other_device(
    dev: Device, dev_2: Device, dir: int, interface: int, tol: float = 0.1
):
    """
    Glue another device to `dev` along a specified direction and interface.
    `dev` is updated in place, `dev_2` is left untouched.

    Parameters:
    -----------
    dev : Device
        The device to glue onto.
    dev_2 : Device
        The other device to be glued.
    dir : int
        Direction along which to glue the devices (0 for x, 1 for y, 2 for z).
    interface : int
        Interface along which to glue the devices (-1 for negative, 1 for positive).
    tol : float
        Tolerance for aligning the devices.
    """

    if interface not in [-1, 1]:
        raise ValueError("interface must be -1 or 1")

    if dir not in [0, 1, 2]:
        raise ValueError("dir must be 0, 1, or 2")

    if dev_2.max_coupling[dir] == 0:
        raise Exception(
            "The other device doesn't have the coupling in this direction"
        )

    if dev_2.max_coupling[dir] > 1:
        raise Exception(
            "The other device has more than one coupling in this direction"
        + " (max_coupling = " + str(dev_2.max_coupling[dir]) + ")")

    if dev.max_coupling[dir] > 1:
        raise Exception(
            "The current device has more than one coupling in this direction"
            + " (max_coupling = " + str(dev.max_coupling[dir]) + ")"
        )

    for dir_check in [0, 1, 2]:
        if dir_check != dir and dev_2.max_coupling[dir_check] != dev.max_coupling[dir_check]:
            raise Exception(
                "The two devices have different coupling ranges in direction " + str(dir_check)
            )

    # Find the orbitals in dev_2 (at the opposite interface) that are periodically coupled
    # Check in all hamiltonians and overlaps
    orb_opposite = []
    for M in (dev_2.H, dev_2.S):
        for keys in M.keys():
            if keys[dir] == -interface:
                orb_opposite = np.union1d(
                    orb_opposite, np.flatnonzero(M[keys].getnnz(axis=0))
                )
    orb_opposite = orb_opposite.astype(int)

    # Find the atoms corresponding to these orbitals
    at_map = np.repeat(
        np.arange(len(dev_2.lattice["at"])),
        [dev_2.orb_map[k] for k in dev_2.lattice["at"]],
    )
    at_opposite = np.unique(at_map[orb_opposite])

    print("Atom in the opposite interface: ", at_opposite)

    AT_1, AT_2 = _match_interface_atoms(dev, dev_2, dir, interface, at_opposite, tol)

    # Get the orbitals corresponding to these atoms
    orb_1 = get_orb_from_at(AT_1, dev.start_orb_per_at)[None, :]
    orb_2 = get_orb_from_at(AT_2, dev_2.start_orb_per_at)[None, :]

    # Check that orb_1 and orb_2 have the same number of orbitals
    if orb_1.shape[1] != orb_2.shape[1]:
        raise Exception(
            "The two devices have a different number of orbitals for the selected atoms"
        )

    # Iterate over the hamiltonian and overlap matrices of dev_2 and glue them to dev
    _glue_blocks(dev.H, dev_2.H, dir, interface, orb_1, orb_2, verbose=True)
    _glue_blocks(dev.S, dev_2.S, dir, interface, orb_1, orb_2)

    # Glue the lattice
    if interface == 1:
        dev.lattice["coords"] = np.concatenate(
            (
                dev_2.lattice["coords"] + dev.lattice["L"][dir, :],
                dev.lattice["coords"],
            ),
            axis=0,
        )
    else:
        dev.lattice["coords"] = np.concatenate(
            (
                dev_2.lattice["coords"],
                dev.lattice["coords"] + dev_2.lattice["L"][dir, :],
            ),
            axis=0,
        )
    dev.lattice["at"] = np.concatenate(
        (dev_2.lattice["at"], dev.lattice["at"]), axis=0
    )
    dev.lattice["L"][dir, :] += dev_2.lattice["L"][dir, :]

    dev.orb_map.update(dev_2.orb_map)

    # Update orb_per_at
    refresh_orb_per_at(dev)

    # The periodic couplings along dir do not describe the glued device anymore
    for M in (dev.H, dev.S):
        for key in list(M.keys()):
            if key[dir] == -1 or key[dir] == 1:
                del M[key]

    update_max_coupling(dev)

    dev.PBC[dir] = False  # The device is no longer periodic in the direction of the contact


def _glue_blocks(
    M_self: dict,
    M_other: dict,
    dir: int,
    interface: int,
    orb_1: np.ndarray,
    orb_2: np.ndarray,
    verbose: bool = False,
):
    """
    Glue every in-cell block of `M_other` onto the matching block of `M_self`.

    Parameters:
    -----------
    M_self : dict
        Hamiltonian or Overlap dictionary of the device glued onto, updated in place.
    M_other : dict
        The corresponding dictionary of the device being glued.
    dir : int
        Direction along which the devices are glued.
    interface : int
        Interface along which the devices are glued (-1 or 1).
    orb_1, orb_2 : np.ndarray
        Matching interface orbitals of the two devices.
    verbose : bool
        Whether to print the key of every block being glued.
    """
    for key in M_other.keys():
        if key[dir] != 0:
            continue

        coup_key_up = list(key)
        coup_key_up[dir] = -interface
        coup_key_up = tuple(coup_key_up)

        coup_key_down = list(key)
        coup_key_down[dir] = interface
        coup_key_down = tuple(coup_key_down)

        if verbose:
            print("Gluing hamiltonian with key ", key)

        M_self[key] = _glue_matrices(
            M_self[key],
            M_other[key],
            M_other[coup_key_up],
            M_other[coup_key_down],
            orb_1,
            orb_2,
        )


def _match_interface_atoms(
    dev: Device,
    dev_2: Device,
    dir: int,
    interface: int,
    at_opposite: np.ndarray,
    tol: float,
):
    """
    Match the atoms at the interface of `dev_2` with the atoms of `dev`.

    Parameters:
    -----------
    dev : Device
        The device glued onto.
    dev_2 : Device
        The other device to be glued.
    dir : int
        Direction along which the devices are glued.
    interface : int
        Interface along which the devices are glued (-1 or 1).
    at_opposite : np.ndarray
        Indices of the atoms of `dev_2` at the opposite interface.
    tol : float
        Tolerance for matching atom positions.

    Returns:
    --------
    AT_1 : np.ndarray
        Indices of the matching atoms in `dev`.
    AT_2 : np.ndarray
        Indices of the matched atoms in `dev_2`, in the same order.
    """

    AT_2 = []
    AT_1 = []
    # Find the corresponding atoms in dev.lattice
    for at in at_opposite:
        if interface == -1:
            delta = np.linalg.norm(
                dev_2.lattice["coords"][at, :] - dev.lattice["coords"], axis=1
            )
        else:
            delta = np.linalg.norm(
                dev_2.lattice["coords"][at, :]
                - dev_2.lattice["L"][dir, :]
                - dev.lattice["coords"]
                + dev.lattice["L"][dir, :],
                axis=1,
            )

        found = np.nonzero(delta < tol)[0]
        if found.size == 0:
            print(
                "Atom ",
                dev_2.lattice["at"][at],
                ", at position ",
                dev_2.lattice["coords"][at, :],
                " not found in the other device.",
            )
            # TODO EVENTUALLY REMOVE ALL THE COUPLINGS RELATED TO THIS ATOM
        if found.size > 1:
            raise ValueError(
                f"Error in gluing the two devices: "
                f"Multiple atoms found in the other device"
                f"matching atom {at} from the second device."
            )
        if found.size == 1:
            # If is found, check that the atom types are the same, and save the indices for both devices
            if dev_2.lattice["at"][at] != dev.lattice["at"][found[0]]:
                raise ValueError(
                    f"Error in gluing the two devices: "
                    f"Atom type mismatch for atom {at} from the second device."
                )
            AT_2.append(at)
            AT_1.append(found[0])

    AT_1 = np.array(AT_1, dtype=int)
    AT_2 = np.array(AT_2, dtype=int)

    # Check that AT_1 is unique
    if len(AT_1) != len(np.unique(AT_1)):
        raise ValueError(
            "Error in gluing the two devices: "
            "Multiple atoms in the first device match atoms in the second device."
        )
    # Check that AT_2 is unique
    if len(AT_2) != len(np.unique(AT_2)):
        raise ValueError(
            "Error in gluing the two devices: "
            "Multiple atoms in the second device match atoms in the first device."
        )

    return AT_1, AT_2


def _glue_matrices(
    M1: sp.csr_matrix,
    M2: sp.csr_matrix,
    coup_up: sp.csr_matrix,
    coup_down: sp.csr_matrix,
    orb_1: np.ndarray,
    orb_2: np.ndarray,
):
    """
    Glue two matrices M1 and M2 with coupling matrix coup between the orbitals orb_1 and orb_2.

    Parameters:
    -----------
    M1 : sp.csr_matrix
        First matrix to be glued.
    M2 : sp.csr_matrix
        Second matrix to be glued.
    coup_up : sp.csr_matrix
        Coupling matrix from M2 towards M1.
    coup_down : sp.csr_matrix
        Coupling matrix from M1 towards M2.
    orb_1 : np.ndarray
        Orbitals in M1 that are coupled.
    orb_2 : np.ndarray
        Orbitals in M2 that are coupled.
    Returns:
    --------
    M_tot : sp.csr_matrix
        The glued matrix.
    """
    # Cut the coupling matrix to keep only the relevant orbitals
    orb_all_2 = np.arange(M2.shape[0])[None, :]
    coup_cut_up = coup_up[orb_all_2.T, orb_2]
    coup_cut_low = coup_down[orb_2.T, orb_all_2]

    N_2 = M2.shape[0]

    # Build the total matrix
    M_tot = sp.block_diag((M2, M1)).tocsr()

    # Insert the coupling matrices
    orb_1 = orb_1 + N_2
    M_tot[orb_all_2.T, orb_1] = coup_cut_up
    M_tot[orb_1.T, orb_all_2] = coup_cut_low

    return M_tot
