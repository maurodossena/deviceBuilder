import os

import numpy as np
from ase import Atoms
from ase.visualize import view
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.linalg import eigh

import deviceBuilder.utils as utils

factor = 27.2114


class Device:

    def __init__(self):
        """
        Initialize the Device object.
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

        self.coup_available = [
            0,
            0,
            0,
        ]  # List to check if the coupling in x,y,z is available (not folded into 000 Hamiltonian or not available at all)
        self.bs_available = False  # Boolean to check if the band structure can be computed (i.e. if the couplings in all directions are available)

    def visualize_lattice(self):
        """
        Visualize the lattice using ASE's view function.
        """

        # CALL ASE VIEW
        symbols = list(self.lattice["at"])
        positions = self.lattice["coords"]
        cell = self.lattice["L"]

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[1, 1, 1])
        view(atoms)

    def compute_band_structure(self, k_path: list, n_point: int = 10, dE: float = 2.0):
        """
        Compute and plot the band structure along a specified k-point path.

        Parameters:
        -----------
        k_path : list
            List of k-points defining the path in reciprocal space.
        n_point : int
            Number of points to interpolate between each pair of k-points in k_path.
        dE : float
            Energy range around the Fermi level to consider for eigenvalue computation.
        Returns:
        --------
        eig_tot : list
            List of eigenvalues computed at each k-point.
        """

        if not self.bs_available:
            raise Exception(
                "I deleted some elements of the hamiltonian, I can't compute the band structure"
            )

        # Build the k-point list from the k_path list
        k_points = []
        for i in range(len(k_path) - 1):
            k_start = np.array(k_path[i])
            k_end = np.array(k_path[i + 1])

            if k_start[0] != 0 or k_end[0] != 0:
                if self.coup_available[0] == 0:
                    raise Exception("I don't have the coupling in x direction")
            if k_start[1] != 0 or k_end[1] != 0:
                if self.coup_available[1] == 0:
                    raise Exception("I don't have the coupling in y direction")
            if k_start[2] != 0 or k_end[2] != 0:
                if self.coup_available[2] == 0:
                    raise Exception("I don't have the coupling in z direction")

            segment = [
                k_start + (k_end - k_start) * t / n_point for t in range(n_point)
            ]
            k_points.extend(segment)
        k_points.append(k_path[-1])

        k_points = np.array(k_points)
        eig_tot = []

        # Iterate over the k-points and compute the eigenvalues
        for i in range(len(k_points)):
            k = k_points[i]
            print("Computing k-point ", k)
            H_k = sp.csr_matrix(self.H[(0, 0, 0)].shape)
            S_k = sp.csr_matrix(self.S[(0, 0, 0)].shape)

            # Sum over all the Hamiltonian and Overlap matrices with correct phase factor
            for key in self.H.keys():
                phase = np.exp(1j * np.pi * np.dot(k, np.asarray(key)))
                H_k += self.H[key] * phase
            for key in self.S.keys():
                phase = np.exp(1j * np.pi * np.dot(k, np.asarray(key)))
                S_k += self.S[key] * phase

            H_k = H_k.todense()
            S_k = S_k.todense()

            # Compute the eigenvalues around the Fermi level
            eigvals, _ = eigh(
                H_k, b=S_k, subset_by_value=(self.Fermi - dE, self.Fermi + dE)
            )
            # Plot the eigenvalues at that k-point
            plt.scatter([i] * len(eigvals), eigvals, color="black", s=1)

            # Append the eigenvalues to the total list
            eig_tot.append(eigvals)

        plt.ylim(self.Fermi - dE, self.Fermi + dE)
        plt.show()

        return eig_tot

    def load_from_cp2k_K_POINTS(
        self,
        filename: str,
        path: str = "./",
        symmetrize: bool = True,
        eps: float = 1e-5,
    ):
        """
        Load Hamiltonian and Overlap matrices from CP2K output files with K_POINTS.

        Parameters:
        -----------
        filename : str
            Path to the CP2K OUTPUT file.
        path : str
            Directory where the CP2K output files are located.
        symmetrize : bool
            Whether to symmetrize the Hamiltonian and Overlap matrices.
        """

        # Read the CP2K settings from the output file
        cp2k_settings = utils.read_cp2k_file(path + filename)

        # Extract the orb map
        self.orb_map = cp2k_settings["no_orb"]

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = utils.read_xyz(
            path + cp2k_settings["coordFile"]
        )

        self.Fermi = cp2k_settings["fermi"]

        # Extract the starting orbital index for each atom
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

        # Load the Hamiltonian for all the k-points
        for i in range(cp2k_settings["n_KS"]):
            file_path = (
                path
                + cp2k_settings["project_name"]
                + "-KS_SPIN_1_R_"
                + str(i + 1)
                + "-1_0.csr"
            )
            temp = utils.read_bin(file_path)
            key = cp2k_settings["KS_list"][i]
            print("Reading Hamiltonian ", key)
            if temp.size != 0:
                temp = utils.bin_to_sparse(temp) * factor
                if key == (0, 0, 0):
                    size_mat = temp.shape[0]
                temp = temp.tocoo()
                temp._shape = (size_mat, size_mat)  # Ensure the shape is set correctly
                temp = temp.tocsr()
                if symmetrize:
                    temp = utils.symmetrize_block(temp)
                temp.data[np.absolute(temp.data) <= eps] = 0
                self.H[key] = temp
                print(
                    "Hamiltonian Is not empty, I'm saving it. Size for",
                    key,
                    " is: ",
                    self.H[key].size,
                )

        # Load the Overlap for all the k-points
        for i in range(cp2k_settings["n_S"]):
            file_path = (
                path
                + cp2k_settings["project_name"]
                + "-S_SPIN_1_R_"
                + str(i + 1)
                + "-1_0.csr"
            )
            temp = utils.read_bin(file_path)
            key = cp2k_settings["S_list"][i]
            print("Reading Overlap ", key)
            if temp.size != 0:
                temp = utils.bin_to_sparse(temp)
                temp = temp.tocoo()
                temp._shape = (size_mat, size_mat)  # Ensure the shape is set correctly
                temp = temp.tocsr()
                if symmetrize:
                    temp = utils.symmetrize_block(temp)
                temp.data[np.absolute(temp.data) <= eps * 0.1] = 0
                self.S[key] = temp
                print(
                    "Overlap Is not empty, I'm saving it. Size for",
                    key,
                    " is: ",
                    self.S[key].size,
                )

        # All couplings are available
        self.coup_available = [1, 1, 1]
        self.bs_available = True

    def center_cell(self):
        """
        Center the atomic coordinates in the unit cell.
        """
        coords_rel = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])
        center = (np.min(coords_rel, axis=0) + np.max(coords_rel, axis=0)) / 2
        self.lattice["coords"] -= center @ self.lattice["L"]
        self.lattice["coords"] += np.array([0.5, 0.5, 0.5]) @ self.lattice["L"]

    def _extract_coup_from_gamma(self, M: dict, dir: int):
        """
        Extract the coupling matrices in the specified direction from the Gamma-point Hamiltonian.

        Parameters:
        -----------
        M : dict
            Dictionary containing the Hamiltonian or Overlap matrices. The 'dir' component of the keys should be 0.
        dir : int
            Direction along which to extract the coupling matrices.
        """

        # Compute the fractional coordinates of the atoms
        frac_coord = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])

        # Sort the atoms along the specified direction
        sorted_at = np.argsort(frac_coord[:, dir])

        # Get the sorted orbitals and the indices to go back to the original order
        sorted_orb = utils.get_orb_from_at(sorted_at, self.start_orb_per_at)[None, :]
        sorted_orb_back = np.argsort(sorted_orb.squeeze())[None, :]

        for key in list(M.keys()):
            # Sort the hamiltonian and overlap matrix
            M_S = M[key][sorted_orb.T, sorted_orb]

            if key[dir] != 0:
                raise Exception(
                    "The hamiltonian has already not Gamma couplings in the selected direction"
                )

            # TODO check if the periodic coupling can be obtained

            # Extract the three matrices (cut over the shifted diagonal)
            bnd = int(np.round(np.shape(M_S)[0] / 2)) + 1
            M_3 = utils.get_upper_block(M_S, bnd)
            M_5 = utils.get_lower_block(M_S, -bnd)

            M_4 = M_S - M_3
            M_4 = M_4 - M_5
            M_4.eliminate_zeros()

            # Reorder the matrices to the original order
            M_3 = M_3[sorted_orb_back.T, sorted_orb_back]
            M_4 = M_4[sorted_orb_back.T, sorted_orb_back]
            M_5 = M_5[sorted_orb_back.T, sorted_orb_back]

            key_l = list(key)

            # Update the dictionaries
            M[key] = M_4

            key_l[dir] = 1
            M[tuple(key_l)] = M_5

            key_l[dir] = -1
            M[tuple(key_l)] = M_3

    def glue_other_device(
        self, dev_2: "Device", dir: int, interface: int, tol: float = 0.1
    ):
        """
        Glue another device to the current device along a specified direction and interface.

        Parameters:
        -----------
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

        if dev_2.coup_available[dir] == 0:
            raise Exception(
                "The other device doesn't have the coupling in this direction"
            )

        if (
            np.abs(
                np.min(self.lattice["coords"][:, dir])
                - np.min(dev_2.lattice["coords"][:, dir])
            )
            > 5 * tol
        ):
            raise Exception("The two devices are not aligned in the selected direction")

        # TODO CHECK THAT THERE IS ONLY ONE COUPLING IN THE DIRECTION dir

        # Find the orbitals in dev_2 (at the opposite interface) that are periodically coupled
        # Check in all hamiltonians and overlaps
        orb_opposite = []
        for keys in dev_2.H.keys():
            if keys[dir] == -interface:
                orb_opposite = np.union1d(
                    orb_opposite, np.flatnonzero(dev_2.H[keys].getnnz(axis=0))
                )
        for keys in dev_2.S.keys():
            if keys[dir] == -interface:
                orb_opposite = np.union1d(
                    orb_opposite, np.flatnonzero(dev_2.S[keys].getnnz(axis=0))
                )
        orb_opposite = orb_opposite.astype(int)

        # Find the atoms corresponding to these orbitals
        at_map = np.repeat(
            np.arange(len(dev_2.lattice["at"])),
            [dev_2.orb_map[k] for k in dev_2.lattice["at"]],
        )
        at_opposite = np.unique(at_map[orb_opposite])

        AT_2 = []
        AT_1 = []
        # Find the corresponding atoms in self.lattice
        for at in at_opposite:
            if interface == -1:
                delta = np.linalg.norm(
                    dev_2.lattice["coords"][at, :] - self.lattice["coords"], axis=1
                )
            else:
                delta = np.linalg.norm(
                    dev_2.lattice["coords"][at, :]
                    - dev_2.lattice["L"][dir, :]
                    - self.lattice["coords"]
                    + self.lattice["L"][dir, :],
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
                if dev_2.lattice["at"][at] != self.lattice["at"][found[0]]:
                    raise ValueError(
                        f"Error in gluing the two devices: "
                        f"Atom type mismatch for atom {at} from the second device."
                    )
                AT_2.append(at)
                AT_1.append(found[0])

        AT_1 = np.array(AT_1, dtype=int)
        AT_2 = np.array(AT_2, dtype=int)

        # Get the orbitals corresponding to these atoms
        orb_1 = utils.get_orb_from_at(AT_1, self.start_orb_per_at)[None, :]
        orb_2 = utils.get_orb_from_at(AT_2, dev_2.start_orb_per_at)[None, :]

        # Check that orb_1 and orb_2 have the same number of orbitals
        if orb_1.shape[1] != orb_2.shape[1]:
            raise Exception(
                "The two devices have a different number of orbitals for the selected atoms"
            )

        # Iterate over the hamiltonian and overlap matrices of dev_2 and glue them to self
        for key in dev_2.H.keys():
            if key[dir] == 0:
                coup_key = list(key)
                coup_key[dir] = -interface
                coup_key = tuple(coup_key)
                print(
                    "Gluing hamiltonian with key ", key, " and coupling key ", coup_key
                )
                self.H[key] = self._glue_matrices(
                    self.H[key], dev_2.H[key], dev_2.H[coup_key], orb_1, orb_2
                )
        for key in dev_2.S.keys():
            if key[dir] == 0:
                coup_key = list(key)
                coup_key[dir] = -interface
                coup_key = tuple(coup_key)
                self.S[key] = self._glue_matrices(
                    self.S[key], dev_2.S[key], dev_2.S[coup_key], orb_1, orb_2
                )

        # Glue the lattice
        if interface == 1:
            self.lattice["coords"] = np.concatenate(
                (
                    dev_2.lattice["coords"] + self.lattice["L"][dir, :],
                    self.lattice["coords"],
                ),
                axis=0,
            )
            self.lattice["at"] = np.concatenate(
                (dev_2.lattice["at"], self.lattice["at"]), axis=0
            )
        else:
            self.lattice["coords"] = np.concatenate(
                (
                    dev_2.lattice["coords"],
                    self.lattice["coords"] + dev_2.lattice["L"][dir, :],
                ),
                axis=0,
            )
            self.lattice["at"] = np.concatenate(
                (dev_2.lattice["at"], self.lattice["at"]), axis=0
            )
        self.lattice["L"][dir, :] += dev_2.lattice["L"][dir, :]

        # Update orb_per_at
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

    def _glue_matrices(
        self,
        M1: sp.csr_matrix,
        M2: sp.csr_matrix,
        coup: sp.csr_matrix,
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
        coup : sp.csr_matrix
            Coupling matrix between M1 and M2.
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
        coup_cut = coup[orb_all_2.T, orb_2]

        N_2 = M2.shape[0]

        # Build the total matrix
        M_tot = sp.block_diag((M2, M1)).tocsr()

        # Insert the coupling matrices
        orb_1 = orb_1 + N_2
        M_tot[orb_all_2.T, orb_1] = coup_cut
        M_tot[orb_1.T, orb_all_2] = coup_cut.T

        return M_tot

    def load_from_cp2k_GAMMA(
        self,
        filename: str,
        path: str = "./",
        extract_x: bool = True,
        extract_y: bool = False,
        extract_z: bool = False,
        symmetrize: bool = True,
        eps: float = 1e-5,
    ):
        """
        Load Hamiltonian and Overlap matrices from CP2K output files at the Gamma point,
        and extract coupling matrices in specified directions.

        Parameters:
        -----------
        path : str
            Directory where the CP2K output files are located.
        filename : str
            Path to the CP2K OUTPUT file.
        extract_x : bool
            Whether to extract coupling matrices in the x direction.
        extract_y : bool
            Whether to extract coupling matrices in the y direction.
        extract_z : bool
            Whether to extract coupling matrices in the z direction.
        symmetrize : bool
            Whether to symmetrize the Hamiltonian and Overlap matrices.
        eps : float
            Threshold below which matrix elements are set to zero.
        """

        # Extract only x coupling
        if extract_x:
            self.coup_available[0] = 1
            self.bs_available = True
        if extract_y:
            self.coup_available[1] = 1
            self.bs_available = True
        if extract_z:
            self.coup_available[2] = 1
            self.bs_available = True

        # Read the CP2K settings from the output file
        cp2k_settings = utils.read_cp2k_file(path + "/" + filename)

        # Extract the orb map
        self.orb_map = cp2k_settings["no_orb"]

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = utils.read_xyz(
            path + "/" + cp2k_settings["coordFile"]
        )

        # Extract the Fermi level
        self.Fermi = cp2k_settings["fermi"]
        # Extract the starting orbital index for each atom
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

        # Read the KS file (GAMMA)
        KS = (
            utils.bin_to_sparse(utils.read_bin(path + "/" + cp2k_settings["KSfile"]))
            * factor
        )
        S = utils.bin_to_sparse(utils.read_bin(path + "/" + cp2k_settings["Sfile"]))

        # Symmetrize the matrices
        if symmetrize:
            KS = utils.symmetrize_block(KS)
            S = utils.symmetrize_block(S)

        # Remove elements below the threshold
        KS.data[np.absolute(KS.data) <= eps] = 0
        KS.eliminate_zeros()
        S.data[np.absolute(S.data) <= eps * 0.1] = 0
        S.eliminate_zeros()

        # Save the Gamma matrices
        self.H[(0, 0, 0)] = KS.tocsr()
        self.S[(0, 0, 0)] = S.tocsr()

        # Extract the coupling matrices in the specified directions
        if extract_x:
            self._extract_coup_from_gamma(self.H, 0)
            self._extract_coup_from_gamma(self.S, 0)
        if extract_y:
            self._extract_coup_from_gamma(self.H, 1)
            self._extract_coup_from_gamma(self.S, 1)
        if extract_z:
            self._extract_coup_from_gamma(self.H, 2)
            self._extract_coup_from_gamma(self.S, 2)

    def _reorder_atoms(
        self,
        at_inside_or: np.ndarray,
        at_inside_rep: np.ndarray,
        cont_vec: np.ndarray,
        list_rep: np.ndarray,
        tol: float = 0.1,
    ):
        """
        Reorder atoms in a periodic repetition to match the order in the origin cell.
        This method ensures that the atoms in a periodic repetition of the contact
        unit cell are ordered in the same way as the atoms in the origin cell.

        Parameters
        ----------
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
        coords_rep = self.lattice["coords"][at_inside_rep, :] - cont_vec @ list_rep
        element_rep = self.lattice["at"][at_inside_rep]
        for at in at_inside_or:
            # Find the atoms in the periodic repetition that are close
            # to the atom in the origin cell and have the same element
            delta = coords_rep - self.lattice["coords"][at, :]
            found = np.nonzero(
                (np.linalg.norm(delta, axis=1) < tol)
                & (self.lattice["at"][at] == element_rep)
            )[0]
            if found.size == 0:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Atom {at} not found in the periodic repetition"
                )
            if found.size > 1:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Multiple atoms found in the periodic repetition"
                    f"matching atom {at} from the origin cell."
                )
            # Append the index of the found atom to the sorted list
            sorted.append(at_inside_rep[found[0]])

        return np.array(sorted, dtype=int)

    def _get_atoms_inside_cell(
        self, origin: np.ndarray, vectors: np.ndarray, nx: int, ny: int, nz: int
    ):
        """Gets the indices of atoms inside a specific periodic repetition.

        This method finds all device atoms that fall within the
        specified periodic repetition of the contact unit cell.

        Parameters
        ----------
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
        relative_coords = self.lattice["coords"] - origin

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

    def upscale_cont(
        self,
        orig: np.ndarray,
        vec: np.ndarray,
        dir: int,
        n: int,
        compute_band: bool = False,
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
        compute_band : bool
            Whether to compute and plot the band structure after upscaling (Bs is computed at transverse Gamma).
        """

        def upscale_contact_matrix(M: dict):
            """
            Upscale the contact Hamiltonian or Overlap matrix by adding n periodic repetitions along the specified direction.
            Parameters:
            M : dict
                Dictionary containing the Hamiltonian or Overlap matrices.
            Returns:
            M_or_K : sp.csr_matrix
                The transverse sum of all Hamiltonian or Overlap matrix for the origin cell (for computing the bs).
            M_coup_K : sp.csr_matrix
                The transverse sum of all coupling matrix between the origin cell and the first periodic repetition. (for computing the bs).
            """

            M_or_K = sp.csr_matrix((orb_inside_cont.shape[1], orb_inside_cont.shape[1]))
            M_coup_K = sp.csr_matrix(
                (orb_inside_cont.shape[1], orb_inside_cont.shape[1])
            )

            for key in M.keys():

                # Extract the matrix coupling the origin cell to itself
                M_or = M[key][orb_inside_cont.T, orb_inside_cont]
                # Sum over all the matrices coupling the origin cell to itself
                M_or_K += M_or

                repet = np.array([0, 0, 0])
                repet[dir] = 1

                # Extract the matrix coupling the origin cell to the first periodic repetition
                at_inside_rep = self._get_atoms_inside_cell(
                    orig, vec, repet[0], repet[1], repet[2]
                )
                at_inside_rep = self._reorder_atoms(
                    at_inside_cont, at_inside_rep, vec, repet
                )
                orb_inside_rep = utils.get_orb_from_at(
                    at_inside_rep, self.start_orb_per_at
                )[None, :]

                # Extract the matrix coupling the origin cell to the first periodic repetition
                M_coup = M[key][orb_inside_cont.T, orb_inside_rep]
                # Sum over all the matrices coupling the origin cell to the first periodic repetition
                M_coup_K += M_coup

                # Check that there are no couplings outside the unit cell and the first periodic repetition
                # Get all the orbitals not in the origin cell and the first periodic repetition
                orb_extra = np.setdiff1d(
                    np.arange(M[key].shape[0]),
                    np.concatenate((orb_inside_cont, orb_inside_rep)),
                )[None, :]

                M_extra = M[key][orb_inside_cont.T, orb_extra]
                if M_extra.nnz != 0:
                    raise Exception(
                        "The contact hamiltonian has couplings outside the unit cell and the first periodic repetition"
                    )

                orb_last_added = orb_inside_cont.copy()
                # Iteratively add n periodic repetitions
                for i in range(n):
                    orb_new = np.arange(
                        M[key].shape[0], M[key].shape[0] + M_or.shape[0]
                    )[None, :]
                    M[key] = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    M[key],
                                    sp.csr_matrix((M[key].shape[0], M_or.shape[1])),
                                )
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

        # Get the atoms and orbitals inside the origin cell
        at_inside_cont = self._get_atoms_inside_cell(orig, vec, 0, 0, 0)
        orb_inside_cont = utils.get_orb_from_at(at_inside_cont, self.start_orb_per_at)[
            None, :
        ]

        # Remove the periodic couplings in the contact directions
        key_c = list(self.H.keys())
        for key in key_c:
            if key[dir] != 0:
                del self.H[key]
        key_c = list(self.S.keys())
        for key in key_c:
            if key[dir] != 0:
                del self.S[key]

        # Upscale the Hamiltonian and Overlap matrices
        H_or_K, H_coup_K = upscale_contact_matrix(self.H)
        S_or_K, S_coup_K = upscale_contact_matrix(self.S)

        # Add the atoms and orbitals of the n periodic repetitions to the device
        for i in range(n):
            self.lattice["coords"] = np.concatenate(
                (
                    self.lattice["coords"],
                    self.lattice["coords"][at_inside_cont, :] - (i + 1) * vec[dir, :],
                ),
                axis=0,
            )
            self.lattice["at"] = np.concatenate(
                (self.lattice["at"], self.lattice["at"][at_inside_cont]), axis=0
            )

            self.lattice["L"][dir, :] += vec[dir, :]

        # Compute and plot the band structure at transverse Gamma
        if compute_band:

            for k in np.linspace(-1, 1, 20):

                H_k = (
                    H_or_K
                    + H_coup_K * np.exp(1j * np.pi * k)
                    + H_coup_K.T * np.exp(-1j * np.pi * k)
                )
                S_k = (
                    S_or_K
                    + S_coup_K * np.exp(1j * np.pi * k)
                    + S_coup_K.T * np.exp(-1j * np.pi * k)
                )

                H_k = H_k.todense()
                S_k = S_k.todense()

                eigvals, _ = eigh(
                    H_k, b=S_k, subset_by_value=(self.Fermi - 2, self.Fermi + 2)
                )
                plt.scatter([k] * len(eigvals), eigvals, color="black", s=1)

            plt.ylim(self.Fermi - 2, self.Fermi + 2)
            plt.show()

        # Update orb_per_at
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

        self.bs_available = False
        self.coup_available[dir] = 0

    def _upscale_mat(self, M: dict, n: int, dir: int):
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

        return M_new

    def upscale(self, n: int, dir: int = 0):
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

        if self.coup_available[dir] == 0:
            raise Exception("I don't have the coupling in this direction")

        # Upscale the Hamiltonian and Overlap matrices
        self.H = self._upscale_mat(self.H, n, dir)
        self.S = self._upscale_mat(self.S, n, dir)

        # Upscale the atoms and orbitals of the n periodic repetitions to the device
        self.lattice["coords"] = np.concatenate(
            [self.lattice["coords"]]
            + [
                self.lattice["coords"] + self.lattice["L"][dir, :] * i
                for i in range(1, n)
            ],
            axis=0,
        )

        # Upscale the atom types
        self.lattice["at"] = np.concatenate(
            [self.lattice["at"]] + [self.lattice["at"] for _ in range(1, n)], axis=0
        )

        # Upscale the lattice vector
        self.lattice["L"][dir, :] *= n

        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

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
        rem_at = utils.find_in_lattice(self.lattice["coords"], b1, b2)
        rem_orb = utils.get_orb_from_at(rem_at, self.start_orb_per_at)

        # Get the atoms and orbitals to keep
        tot_at = np.arange(self.lattice["coords"].shape[0])
        tot_orb = np.arange(self.H[0].shape[0])
        keep_at = np.setdiff1d(tot_at, rem_at)
        keep_orb = np.setdiff1d(tot_orb, rem_orb)[None, :]

        # Keep only the orbitals in the hamiltonian
        for i in range(len(self.H)):
            self.H[i] = self.H[i][keep_orb.T, keep_orb]
            self.S[i] = self.S[i][keep_orb.T, keep_orb]
        # Keep only the atoms in the lattice
        self.lattice["coords"] = self.lattice["coords"][keep_at, :]
        self.lattice["at"] = self.lattice["at"][keep_at]
        # Update orb_per_at
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )

    def generate_potential_barrier(self, V1, V2, V3, slope, dir):
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

        from scipy.special import erf

        if dir not in [0, 1, 2]:
            raise ValueError("dir must be 0, 1, or 2")

        # Define the positions of the barriers
        l_s = self.lattice["L"][dir, dir] / 3
        l1 = np.min(self.lattice["coords"][:, dir]) + l_s
        l2 = l1 + l_s

        # Generate the potential
        self.potential = np.ones(self.lattice["coords"].shape[0]) * V1
        self.potential += (V2 - V1) * (
            0.5 * erf((self.lattice["coords"][:, dir] - l1) / slope) + 0.5
        )
        self.potential += (V3 - V2) * (
            0.5 * erf((self.lattice["coords"][:, dir] - l2) / slope) + 0.5
        )

    def export_data(self, transport_dir=[0], output_dir="device/inputs"):
        """
        Export the device data to a specified directory in the QUATREX format.

        Parameters:
        -----------
        transport_dir : list
            List of directions along which transport occurs (0 for x, 1 for y, 2 for z).
        output_dir : str
            Directory where the device data will be saved.
        """

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save Hamiltonian and Overlap matrices
        for key, value in self.H.items():
            # Don't save the coupling matrices in the transport directions
            check = True
            for dir in transport_dir:
                if key[dir] != 0:
                    check = False
                    break
            if check:
                sp.save_npz(
                    os.path.join(
                        output_dir, f"hamiltonian_{key[0]}_{key[1]}_{key[2]}.npz"
                    ),
                    value.tocsr(),
                )

        for key, value in self.S.items():
            # Don't save the coupling matrices in the transport directions
            check = True
            for dir in transport_dir:
                if key[dir] != 0:
                    check = False
                    break
            if check:
                sp.save_npz(
                    os.path.join(output_dir, f"overlap_{key[0]}_{key[1]}_{key[2]}.npz"),
                    value.tocsr(),
                )

        # Create potential.npy if potential is defined
        if self.potential is not None:
            np.save(os.path.join(output_dir, "potential.npy"), self.potential)

        # Crate lattice.xyz
        xyz_path = os.path.join(output_dir, "lattice.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(self.lattice['at'])}\n")  # first line: number of atoms
            lattice_vals = " ".join(f"{val:.6f}" for val in self.lattice["L"].flatten())
            f.write(f'Lattice="{lattice_vals}"\n')
            for at_type, (x, y, z) in zip(self.lattice["at"], self.lattice["coords"]):
                f.write(f"{at_type} {x:.6f} {y:.6f} {z:.6f}\n")
