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

    def wrap_cell(self):
        """
        Wrap the atomic coordinates into the unit cell defined by the lattice vectors.
        This method ensures that all atomic coordinates are within the bounds of the unit cell.
        """

        coords_rel = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])
        translation = -np.floor(coords_rel)
        
        for i in range(3):
            if not self.PBC[i]:
                translation[:, i] = 0

        self.lattice["coords"] += translation @ self.lattice["L"]

        # If there is only one matrix in a direction, the coupling
        # is assumed to be summed over all the periodic image,
        # so the translation in that direction is not relevant for the 
        # matricees.
        for i in range(3):
            if self.max_coupling[i] == 0:
                translation[:,i] = 0

        unique_translations, counts = np.unique(translation, axis=0, return_counts=True)
        ref_translation = unique_translations[np.argmax(counts)]
        translations_hamiltonian = translation - ref_translation

        unique_translations_hamiltonian = unique_translations - ref_translation

        for t in unique_translations_hamiltonian:

            at_t = np.where(np.all(translations_hamiltonian == t, axis=1))[0]
            orb_t = get_orb_from_at(at_t, self.start_orb_per_at)

            self._wrap_matrix("H", t, orb_t)
            self._wrap_matrix("S", t, orb_t)

        self._update_max_coupling()

        self._add_missing_couplings(self.H[(0,0,0)].shape[0], "H")
        self._add_missing_couplings(self.S[(0,0,0)].shape[0], "S")

        self._check_if_hermitian("H")
        self._check_if_hermitian("S")

    def _wrap_matrix(self, matrix_type, t, orb):

        def swap_elements(A,B,C, elements):
            A_c, B_c, C_c = A.tocoo(), B.tocoo(), C.tocoo()

            mask = np.zeros(B_c.shape[0], dtype=bool)
            mask[elements] = True
            nmask = ~mask

            B_keep = (mask[B_c.row] & mask[B_c.col]) | (nmask[B_c.row] & nmask[B_c.col])
            A_keep = nmask[A_c.row] & mask[A_c.col]
            C_keep = mask[C_c.row] & nmask[C_c.col]

            row = np.concatenate([A_c.row[A_keep], B_c.row[B_keep], C_c.row[C_keep]])
            col = np.concatenate([A_c.col[A_keep], B_c.col[B_keep], C_c.col[C_keep]])
            data = np.concatenate([A_c.data[A_keep], B_c.data[B_keep], C_c.data[C_keep]])

            shape = A_c.shape
            return sp.csr_matrix((data, (row, col)), shape=shape)
        
        def shift_key(key, t):
            return tuple((np.int32(np.array(key) + t)).tolist())
        
        if matrix_type == "H":
            M = self.H
        elif matrix_type == "S":
            M = self.S
        else:
            raise ValueError("matrix_type must be 'H' or 'S'")

        M_wrap={}

        if np.array_equal(t, [0, 0, 0]):
            return

        # The elements coupling a translated atom to a non translated one end up in
        # a different cell, so the destination keys have to be visited as well: an
        # element landing on a key that is not stored (even only in an intermediate
        # pass, before another translation brings it back in range) would be lost.
        shape = M[(0, 0, 0)].shape
        keys_H = set(M.keys())
        keys_H |= {shift_key(key, t) for key in M.keys()}
        keys_H |= {shift_key(key, -t) for key in M.keys()}

        empty = sp.csr_matrix(shape)

        for key in keys_H:
            key_m1 = shift_key(key, t)
            key_p1 = shift_key(key, -t)

            H_m1 = M.get(key_m1, empty)
            H_p1 = M.get(key_p1, empty)

            M_wrap[key] = swap_elements(H_m1, M.get(key, empty), H_p1, orb)

        # Drop the keys that stayed empty, so that the range of the couplings does
        # not grow pass after pass.
        if matrix_type == "H":
            self.H = {k: mm for k, mm in M_wrap.items() if mm.nnz != 0 or k == (0, 0, 0)}
        else:
            self.S = {k: mm for k, mm in M_wrap.items() if mm.nnz != 0 or k == (0, 0, 0)}


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
