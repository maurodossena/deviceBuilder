"""Transformations of the lattice and of the on-site energies of a device."""

import numpy as np
from scipy import sparse as sp

from .base import DeviceBase
from .lattice import find_in_lattice, get_orb_from_at


class TransformMixin(DeviceBase):
    """Rigid motions, sorting, atom removal and energy shifts."""

    def shift_energies(self, dE: float):
        """
        Shift the energies of the Hamiltonian by a specified amount.

        Parameters:
        -----------
        dE : float
            Amount by which to shift the energies (in eV).
        """

        for key in self.H.keys():
            if key in self.S.keys():
                self.H[key] = self.H[key] + self.S[key] * dE
            else:
                self.H[key] = self.H[key] + sp.eye(self.H[key].shape[0]) * dE

        self.Fermi += dE

    def reduce_to_gamma(self, dir: int = 0):
        """
        Reduce the Hamiltonian and Overlap matrices to only the Gamma-point components.
        This method removes all non-Gamma components from the Hamiltonian and Overlap matrices,
        effectively folding the band structure into the Gamma point.
        """

        for M in (self.H, self.S):
            for key in list(M.keys()):
                key_dir = key[dir]
                key_t = list(key)
                del key_t[dir]
                if key_t != [0, 0]:
                    key_target = [0, 0]
                    key_target.insert(dir, key_dir)
                    M[tuple(key_target)] += M[key]
                    del M[key]

        self.coup_available = [0, 0, 0]
        self.coup_available[dir] = 1
        self.bs_available = True

    def center_cell(self):
        """
        Center the atomic coordinates in the unit cell.
        """
        coords_rel = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])
        center = (np.min(coords_rel, axis=0) + np.max(coords_rel, axis=0)) / 2
        self.lattice["coords"] -= center @ self.lattice["L"]
        self.lattice["coords"] += np.array([0.5, 0.5, 0.5]) @ self.lattice["L"]

    def align_with_axis(self):
        """
        Align the atomic coordinates with the lattice vectors.
        """
        coords_rel = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])

        min_rel = np.min(coords_rel, axis=0)

        self.lattice["coords"] -= min_rel @ self.lattice["L"]
        self.lattice["coords"] = np.absolute(self.lattice["coords"])

    def translate_cell(self, shift: np.ndarray):
        """
        Translate the atomic coordinates by a specified shift.

        Parameters:
        -----------
        shift : np.ndarray
            1D array of length 3 specifying the translation vector in Cartesian coordinates.
        """
        if shift.shape != (3,):
            raise ValueError("shift must be a 1D array of length 3")

        self.lattice["coords"] += shift

    def sort_atoms(self, dir: list = [0, 1, 2], decimals: int = 6):
        """
        Sort the atoms and orbitals along a specified direction.
        It will internally use lexsort to sort along multiple directions.

        Parameters:
        -----------
        dir : list
            List of directions along which to sort the atoms (0 for x, 1 for y, 2 for z).
        decimals : int
            Number of decimals to round the fractional coordinates for sorting.
        """
        if not all(d in [0, 1, 2] for d in dir):
            raise ValueError("dir must be a list of 0, 1, or 2")

        # Compute the fractional coordinates of the atoms
        frac_coord = np.round(self.lattice["coords"], decimals) @ np.linalg.inv(
            self.lattice["L"]
        )

        # Sort the atoms along the specified direction
        sorted_at = np.lexsort((frac_coord[:, np.array(dir[::-1]).T]).T)

        # Get the sorted orbitals and the indices to go back to the original order
        sorted_orb = get_orb_from_at(sorted_at, self.start_orb_per_at)[None, :]

        # Reorder the lattice
        self.lattice["coords"] = self.lattice["coords"][sorted_at, :]
        self.lattice["at"] = self.lattice["at"][sorted_at]

        # Reorder the Hamiltonian and Overlap matrices
        for M in (self.H, self.S):
            for key in list(M.keys()):
                M[key] = M[key][sorted_orb.T, sorted_orb]
                M[key].sort_indices()

        # Update the starting orbital index for each atom
        self._refresh_orb_per_at()

    def remove_atoms(self, b1: np.ndarray, b2: np.ndarray):
        """
        Remove atoms within a specified box defined by two corner points b1 and b2.

        Parameters:
        -----------
        b1 : np.ndarray
            The first corner point of the box.
        b2 : np.ndarray
            The second corner point of the box.
        """

        # Find the atoms to remove
        rem_at = find_in_lattice(self.lattice["coords"], b1, b2)
        rem_orb = get_orb_from_at(rem_at, self.start_orb_per_at)

        # Get the atoms and orbitals to keep
        tot_at = np.arange(self.lattice["coords"].shape[0])
        tot_orb = np.arange(self.H[(0, 0, 0)].shape[0])
        keep_at = np.setdiff1d(tot_at, rem_at)
        keep_orb = np.setdiff1d(tot_orb, rem_orb)[None, :]

        # Keep only the orbitals in the hamiltonian
        for M in (self.H, self.S):
            for key in M.keys():
                M[key] = M[key][keep_orb.T, keep_orb]
        # Keep only the atoms in the lattice
        self.lattice["coords"] = self.lattice["coords"][keep_at, :]
        self.lattice["at"] = self.lattice["at"][keep_at]
        # Update orb_per_at
        self._refresh_orb_per_at()

    def generate_potential_barrier(self, V1, V2, V3, slope, dir, grid_from_OMEN=None):
        """
        Generate a smooth potential barrier along a specified direction using error functions.

        Parameters:
        -----------
        V1 : float
            Potential value in the first region.
        V2 : float
            Potential value in the second region.
        V3 : float
            Potential value in the third region.
        slope : float
            Slope of the transition between regions.
        dir : int
            Direction along which to generate the potential (0 for x, 1 for y, 2 for z).
        """

        coords = self.lattice["coords"].copy()

        if grid_from_OMEN is not None:
            print("Using the grid from OMEN")
            coords = grid_from_OMEN.copy()

        from scipy.special import erf

        if dir not in [0, 1, 2]:
            raise ValueError("dir must be 0, 1, or 2")

        # Define the positions of the barriers
        l_s = self.lattice["L"][dir, dir] / 3
        l1 = np.min(self.lattice["coords"][:, dir]) + l_s
        l2 = l1 + l_s

        # Generate the potential
        self.potential = np.ones(coords.shape[0]) * V1
        self.potential += (V2 - V1) * (0.5 * erf((coords[:, dir] - l1) / slope) + 0.5)
        self.potential += (V3 - V2) * (0.5 * erf((coords[:, dir] - l2) / slope) + 0.5)
