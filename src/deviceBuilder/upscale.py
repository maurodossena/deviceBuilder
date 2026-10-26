"""Periodic repetition of a device, and of one of its contacts."""

import numpy as np
from scipy import sparse as sp

from .device import Device, refresh_orb_per_at, update_max_coupling
from .lattice import get_orb_from_at


def upscale(dev: Device, n: int, dir: int = 0):
    """
    Upscale the Hamiltonian and overlap matrix in the specified direction.

    Parameters:
    -----------
    n : int
        Number of repetitions in the specified direction.
    dir : int
        Direction to upscale (0 for x, 1 for y, 2 for z).
    """

    if dir not in [0, 1, 2]:
        raise ValueError("dir must be 0, 1, or 2")

    if dev.max_coupling[dir] == 0:
        raise Exception("I don't have the coupling in this direction")

    # Upscale the Hamiltonian and Overlap matrices
    dev.H = _upscale_mat(dev.H, n, dir)
    dev.S = _upscale_mat(dev.S, n, dir)

    # Upscale the atoms and orbitals of the n periodic repetitions to the device
    dev.lattice["coords"] = np.concatenate(
        [dev.lattice["coords"]]
        + [
            dev.lattice["coords"] + dev.lattice["L"][dir, :] * i
            for i in range(1, n)
        ],
        axis=0,
    )

    # Upscale the atom types
    dev.lattice["at"] = np.concatenate(
        [dev.lattice["at"]] + [dev.lattice["at"] for _ in range(1, n)], axis=0
    )

    # Upscale the lattice vector
    dev.lattice["L"][dir, :] *= n

    refresh_orb_per_at(dev)
    update_max_coupling(dev)


def _upscale_mat(M: dict, n: int, dir: int):
    """
    Upscale the Hamiltonian or Overlap matrix by adding n periodic repetitions along the specified direction.

    Parameters:
    -----------
    M : dict
        Dictionary containing the Hamiltonian or Overlap matrices.
    n : int
        Number of repetitions in the specified direction.
    dir : int
        Direction along which to upscale (0 for x, 1 for y, 2 for z).

    Returns:
    --------
    M_new : dict
        Dictionary containing the upscaled Hamiltonian or Overlap matrices.
    """

    # Get the indices of the other two directions
    other_dirs = [i for i in [0, 1, 2] if i != dir]
    perp_couplings = {(key[other_dirs[0]], key[other_dirs[1]]) for key in M.keys()}

    M_new = {}
    size_mat = M[(0, 0, 0)].shape[0]

    # Iterate over all the perpendicular couplings
    for p_c in perp_couplings:

        M_list = []

        # Get the indices along the direction to upscale
        indices_along_dir = [
            key[dir]
            for key in M.keys()
            if (key[other_dirs[0]], key[other_dirs[1]]) == p_c
        ]
        min_coupling_idx = min(indices_along_dir)
        max_coupling_idx = max(indices_along_dir)

        # Find the padding needed to have the correct size
        n_pad_minus = min_coupling_idx % n
        n_min = (abs(min_coupling_idx) + n_pad_minus) // n
        n_pad_plus = -(abs(max_coupling_idx) + n) % n
        n_plus = (abs(max_coupling_idx) + n + n_pad_plus) // n

        # Start by padding with empty matrices
        for i in range(n_pad_minus):
            M_list.append(sp.csr_matrix((size_mat, size_mat)))

        # Add all the matrices along the direction to upscale
        for i in range(min_coupling_idx, max_coupling_idx + 1):
            key = list(p_c).copy()
            key.insert(dir, i)
            key = tuple(key)
            if key not in M:
                M_list.append(sp.csr_matrix((size_mat, size_mat)))
            else:
                M_list.append(M[key])

        # End by padding with empty matrices
        for i in range(n_pad_plus + n - 1):
            M_list.append(sp.csr_matrix((size_mat, size_mat)))

        M_temp = sp.hstack(M_list, format="csr")

        for i in range(n - 1):
            M_list.pop()
            M_list.insert(0, sp.csr_matrix((size_mat, size_mat)))

            M_temp = sp.vstack((M_temp, sp.hstack(M_list)), format="csr")

        # Split the big matrix into the correct number of sub matrices
        slice_indices = np.split(np.arange(M_temp.shape[1]), n_min + n_plus)
        slices = [M_temp[:, indices] for indices in slice_indices]

        # Save the new matrices in the dictionary
        for i in range(-n_min, n_plus):
            key = list(p_c).copy()
            key.insert(dir, i)
            key = tuple(key)

            M_new[key] = slices[i + n_min]

        # Make sure tha every matrix is in canonical form (i.e. the keys are sorted)
        for key in M_new.keys():
            M_new[key].sort_indices()

    return M_new


def upscale_cont(
    dev: Device,
    orig: np.ndarray,
    vec: np.ndarray,
    dir: int,
    n: int,
):
    """
    Upscale the contact unit cell along a specified direction by adding n periodic repetitions.

    Parameters:
    orig : np.ndarray
        1D array representing the origin of the contact unit cell.
    vec : np.ndarray
        3x3 array of contact lattice vectors.
    dir : int
        Direction along which to upscale (0 for x, 1 for y, 2 for z).
    n : int
        Number of periodic repetitions to add.
    """

    # Get the atoms and orbitals inside the origin cell
    at_inside_cont = _get_atoms_inside_cell(dev.lattice["coords"], orig, vec, 0, 0, 0)
    orb_inside_cont = get_orb_from_at(at_inside_cont, dev.start_orb_per_at)[None, :]

    # Remove the periodic couplings in the contact directions
    for M in (dev.H, dev.S):
        for key in list(M.keys()):
            if key[dir] != 0:
                del M[key]

    # Upscale the Hamiltonian and Overlap matrices
    H_or_K, H_coup_K = _upscale_cont_mat(
        dev, dev.H, orig, vec, dir, n, at_inside_cont, orb_inside_cont
    )
    S_or_K, S_coup_K = _upscale_cont_mat(
        dev, dev.S, orig, vec, dir, n, at_inside_cont, orb_inside_cont
    )

    # Add the atoms and orbitals of the n periodic repetitions to the device
    for i in range(n):
        dev.lattice["coords"] = np.concatenate(
            (
                dev.lattice["coords"],
                dev.lattice["coords"][at_inside_cont, :] - (i + 1) * vec[dir, :],
            ),
            axis=0,
        )
        dev.lattice["at"] = np.concatenate(
            (dev.lattice["at"], dev.lattice["at"][at_inside_cont]), axis=0
        )

        dev.lattice["L"][dir, :] += np.absolute(vec[dir, :])

    # Update orb_per_at
    refresh_orb_per_at(dev)

    update_max_coupling(dev)

    dev.PBC[dir] = False # The device is no longer periodic in the direction of the contact


def _upscale_cont_mat(
    dev: Device,
    M: dict,
    orig: np.ndarray,
    vec: np.ndarray,
    dir: int,
    n: int,
    at_inside_cont: np.ndarray,
    orb_inside_cont: np.ndarray,
):
    """
    Upscale the contact Hamiltonian or Overlap matrix by adding n periodic
    repetitions along the specified direction.

    Parameters:
    -----------
    dev : Device
        The device the contact belongs to (its lattice is only read).
    M : dict
        Dictionary containing the Hamiltonian or Overlap matrices, updated in place.
    orig : np.ndarray
        1D array representing the origin of the contact unit cell.
    vec : np.ndarray
        3x3 array of contact lattice vectors.
    dir : int
        Direction along which to upscale (0 for x, 1 for y, 2 for z).
    n : int
        Number of periodic repetitions to add.
    at_inside_cont : np.ndarray
        Indices of the atoms inside the contact origin cell.
    orb_inside_cont : np.ndarray
        Indices of the orbitals inside the contact origin cell.

    Returns:
    --------
    M_or_K : sp.csr_matrix
        The transverse sum of all Hamiltonian or Overlap matrix for the origin cell (for computing the bs).
    M_coup_K : sp.csr_matrix
        The transverse sum of all coupling matrix between the origin cell and the first periodic repetition. (for computing the bs).
    """

    coords = dev.lattice["coords"]

    M_or_K = sp.csr_matrix((orb_inside_cont.shape[1], orb_inside_cont.shape[1]))
    M_coup_K = sp.csr_matrix((orb_inside_cont.shape[1], orb_inside_cont.shape[1]))

    for key in M.keys():

        # Extract the matrix coupling the origin cell to itself
        M_or = M[key][orb_inside_cont.T, orb_inside_cont]
        # Sum over all the matrices coupling the origin cell to itself
        M_or_K += M_or

        repet = np.array([0, 0, 0])
        repet[dir] = 1

        # Extract the matrix coupling the origin cell to the first periodic repetition
        at_inside_rep = _get_atoms_inside_cell(
            coords, orig, vec, repet[0], repet[1], repet[2]
        )
        # Check that the number of atoms is the same as in the origin cell
        if at_inside_rep.size != at_inside_cont.size:
            raise Exception(
                "The number of atoms in the periodic repetition is different from the origin cell"
            )
        at_inside_rep = _reorder_atoms(dev.lattice, at_inside_cont, at_inside_rep, vec, repet)
        orb_inside_rep = get_orb_from_at(at_inside_rep, dev.start_orb_per_at)[
            None, :
        ]

        # Extract the matrix coupling the origin cell to the first periodic repetition
        M_coup = M[key][orb_inside_cont.T, orb_inside_rep]
        # Sum over all the matrices coupling the origin cell to the first periodic repetition
        M_coup_K += M_coup

        repet[dir] = -1
        at_inside_rep_m1 = _get_atoms_inside_cell(
            coords, orig, vec, repet[0], repet[1], repet[2]
        )
        if at_inside_rep_m1.size > 0:
            raise Exception("There are atoms in the negative periodic repetition")

        repet[dir] = 2
        at_inside_rep_2 = _get_atoms_inside_cell(
            coords, orig, vec, repet[0], repet[1], repet[2]
        )
        orb_inside_rep_2 = get_orb_from_at(at_inside_rep_2, dev.start_orb_per_at)[
            None, :
        ]
        # Check that there are no couplings outside the unit cell and the first periodic repetition
        # Get all the orbitals not in the origin cell and the first periodic repetition
        M_extra_2 = M[key][orb_inside_cont.T, orb_inside_rep_2]
        if M_extra_2.nnz != 0:
            raise Exception(
                "The contact hamiltonian has couplings in the second periodic repetition"
            )

        orb_last_added = orb_inside_cont.copy()
        # Iteratively add n periodic repetitions
        for i in range(n):
            orb_new = np.arange(M[key].shape[0], M[key].shape[0] + M_or.shape[0])[
                None, :
            ]
            M[key] = sp.vstack(
                (
                    sp.hstack(
                        (M[key], sp.csr_matrix((M[key].shape[0], M_or.shape[1])))
                    ),
                    sp.hstack(
                        (sp.csr_matrix((M_or.shape[0], M[key].shape[1])), M_or)
                    ),
                )
            )
            M[key][orb_new.T, orb_last_added] = M_coup
            M[key][orb_last_added.T, orb_new] = M_coup.T

            orb_last_added = orb_new.copy()

    return M_or_K, M_coup_K


def _reorder_atoms(
    lattice: dict,
    at_inside_or: np.ndarray,
    at_inside_rep: np.ndarray,
    cont_vec: np.ndarray,
    list_rep: np.ndarray,
    tol: float = 0.1,
):
    """
    Reorder atoms in a periodic repetition to match the order in the origin cell.
    This function ensures that the atoms in a periodic repetition of the contact
    unit cell are ordered in the same way as the atoms in the origin cell.

    Parameters
    ----------
    lattice : dict
        Lattice of the device.
    at_inside_or : NDArray
        1D array of atom indices inside the origin cell.
    at_inside_rep : NDArray
        1D array of atom indices inside the periodic repetition.
    cont_vec : NDArray
        3x3 array of contact lattice vectors.
    list_rep : NDArray
        1D array of indices for the periodic repetition. Example: [1,0,0] for the first repetition in x.
    tol : float
        Tolerance for matching atom positions.

    Returns
    -------
    NDArray
        1D array of reordered atom indices in the periodic repetition.
    """
    sorted = []

    # Shift the coordinates of the atoms in the periodic repetition
    coords_rep = lattice["coords"][at_inside_rep, :] - cont_vec @ list_rep
    element_rep = lattice["at"][at_inside_rep]
    for at in at_inside_or:
        # Find the atoms in the periodic repetition that are close
        # to the atom in the origin cell and have the same element
        delta = coords_rep - lattice["coords"][at, :]
        found = np.nonzero(
            (np.linalg.norm(delta, axis=1) < tol)
            & (lattice["at"][at] == element_rep)
        )[0]
        if found.size == 0:
            raise ValueError(
                f"Error in the contact: "
                f"Atom {at} not found in the periodic repetition"
            )
        if found.size > 1:
            raise ValueError(
                f"Error in the contact: "
                f"Multiple atoms found in the periodic repetition"
                f"matching atom {at} from the origin cell."
            )
        # Append the index of the found atom to the sorted list
        sorted.append(at_inside_rep[found[0]])

    return np.array(sorted, dtype=int)


def _get_atoms_inside_cell(
    coords: np.ndarray, origin: np.ndarray, vectors: np.ndarray, nx: int, ny: int, nz: int
):
    """Gets the indices of atoms inside a specific periodic repetition.

    This function finds all device atoms that fall within the
    specified periodic repetition of the contact unit cell.

    Parameters
    ----------
    coords : ndarray
        Nx3 array of the atomic coordinates of the device.
    origin : ndarray
        1D array representing the origin of the contact unit cell.
    vectors : ndarray
        3x3 array of contact lattice vectors.
    nx : int
        The x-coordinate of the periodic repetition.
    ny : int
        The y-coordinate of the periodic repetition.
    nz : int
        The z-coordinate of the periodic repetition.

    Returns
    -------
    NDArray
        1D array of atom indices that fall within the specified
        periodic repetition.

    """

    # Shift the coordinates of the device atoms to the origin of the
    # contact
    relative_coords = coords - origin

    # Compute the coefficients relative to the contact cell
    coeffs = relative_coords @ np.linalg.inv(vectors)

    # Get the indices of the atoms inside the periodic repetition
    inside_mask = np.nonzero(
        (coeffs[:, 0] >= nx)
        & (coeffs[:, 0] <= nx + 1)
        & (coeffs[:, 1] >= ny)
        & (coeffs[:, 1] <= ny + 1)
        & (coeffs[:, 2] >= nz)
        & (coeffs[:, 2] <= nz + 1)
    )[0]

    return inside_mask
