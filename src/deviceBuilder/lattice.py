"""Helpers to map between atoms, orbitals and regions of the lattice."""

import numpy as np


def get_orb_from_at(at_ind, orb_per_at):
    """
    Get the orbitals of a specific atom

    Parameters
    ----------
    at_ind : int
        Index of the atom.
    orb_per_at : dict
        Dictionary containing the number of orbitals per atom.

    Returns
    -------
    orb : list
        List of orbitals of the atom.

    """
    orb = []
    for i in at_ind:
        orb.append(np.arange(orb_per_at[i], orb_per_at[i + 1]))
    return np.concatenate(orb)


def find_in_lattice(coords: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Find the atoms in a specific region of the lattice

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the atoms.
    c1 : np.ndarray
        Corner 1 of the region.
    c2 : np.ndarray
        Corner 2 of the region.
    Returns
    -------
    vec_atoms : np.ndarray
        Indices of the atoms in the region.
    """

    vec_atoms = coords[:, 0] >= c1[0]
    vec_atoms &= coords[:, 0] < c2[0]
    vec_atoms &= coords[:, 1] >= c1[1]
    vec_atoms &= coords[:, 1] < c2[1]
    vec_atoms &= coords[:, 2] >= c1[2]
    vec_atoms &= coords[:, 2] < c2[2]
    vec_atoms = np.nonzero(vec_atoms)[0]

    return vec_atoms
