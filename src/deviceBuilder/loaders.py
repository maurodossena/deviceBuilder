"""Loading a device from CP2K, Wannier90 or MALOQ output."""

import numpy as np
from scipy import sparse as sp

from . import io
from .base import DeviceBase
from .constants import HARTREE_TO_EV
from .lattice import get_orb_from_at
from .sparse_ops import get_lower_block, get_upper_block, prune_small, symmetrize_block


class LoaderMixin(DeviceBase):
    """Readers that populate `H`, `S` and `lattice` from external files."""

    def load_from_wannier90(
        self,
        filename: str,
        path: str = "./",
        eps: float = 1e-5,
        gather_wannier_centers: bool = False,
    ):
        """
        Load Hamiltonian and Overlap matrices from Wannier90 output files.

        Parameters:
        -----------
        filename : str
            Path to the Wannier90 output file.
        path : str
            Directory where the Wannier90 output files are located.
        eps : float
            Small value to threshold the matrix elements.
        """

        self.lattice = {}
        self.orb_map = {}
        w_centers, self.lattice["L"], self.lattice["coords"], self.lattice["at"] = (
            io.read_wannier_wout(
                path + "/" + filename, return_atom=True, transform_home_cell=True
            )
        )

        if gather_wannier_centers:

            elements = np.unique(self.lattice["at"])

            for at in elements:
                self.orb_map[at] = 0

            grid = np.array(
                np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
            )
            combos = grid.reshape(3, -1).T
            translations = combos @ self.lattice["L"]

            translated_atoms = (
                self.lattice["coords"][:, :, None] + translations.T[None, :, :]
            )

            w_atoms = np.zeros(len(w_centers), dtype=int)
            for i, w in enumerate(w_centers):
                dist = np.min(
                    np.linalg.norm(w[None, :, None] - translated_atoms, axis=1), axis=1
                )
                index = np.argmin(dist)
                w_atoms[i] = index

            sorted_index = np.argsort(w_atoms)
            _, orb_counts = np.unique(w_atoms, return_counts=True)

            for i, count in enumerate(orb_counts):
                at_i = self.lattice["at"][i]
                if self.orb_map[at_i] == 0:
                    self.orb_map[at_i] = count
                else:
                    if self.orb_map[at_i] != count:
                        raise Exception(
                            "Error in gathering the Wannier centers: different number of orbitals found for atom ",
                            at_i,
                        )

        else:
            self.orb_map["X"] = 1

            self.lattice["coords"] = w_centers
            self.lattice["at"] = np.array(["X"] * len(w_centers))

        h_wann = io.read_hr_dat(path + "/" + filename.replace(".wout", "_hr.dat"))

        print(h_wann.shape)

        n_coup_1 = h_wann.shape[0]
        n_coup_2 = h_wann.shape[1]
        n_coup_3 = h_wann.shape[2]

        for i in range(-(n_coup_1 // 2), (n_coup_1 // 2) + 1):
            for j in range(-(n_coup_2 // 2), (n_coup_2 // 2) + 1):
                for k in range(-(n_coup_3 // 2), (n_coup_3 // 2) + 1):
                    temp = h_wann[i, j, k, :, :]
                    if gather_wannier_centers:
                        temp = temp[sorted_index, :][:, sorted_index]
                    temp = prune_small(sp.csr_matrix(temp), eps)
                    if temp.size != 0:
                        self.H[(i, j, k)] = temp
                        print(
                            "Hamiltonian Is not empty, I'm saving it. Size for",
                            (i, j, k),
                            " is: ",
                            self.H[(i, j, k)].size,
                        )

        self.S[(0, 0, 0)] = sp.eye(self.H[(0, 0, 0)].shape[0], format="csr")

        self._refresh_orb_per_at()

        self.Fermi = -999

        # All couplings are available
        self.coup_available = [1, 1, 1]
        self.bs_available = True

    def load_from_cp2k_K_POINTS(
        self,
        filename: str,
        path: str = "./",
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
        cp2k_settings = io.read_cp2k_file(path + "/" + filename)

        # Extract the orb map
        self.orb_map = cp2k_settings["no_orb"]

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = io.read_xyz(
            path + "/" + cp2k_settings["coordFile"]
        )

        # TODO CHECK IF IT IS WRAPPED
        self.lattice["coords"] += self.lattice["L"] @ np.array([0.5, 0.5, 0.5])

        self.Fermi = cp2k_settings["fermi"]

        # Extract the starting orbital index for each atom
        self._refresh_orb_per_at()

        # Load the Hamiltonian for all the k-points
        for i in range(cp2k_settings["n_KS"]):
            file_path = (
                path
                + "/"
                + cp2k_settings["project_name"]
                + "-KS_SPIN_1_R_"
                + str(i + 1)
                + "-1_0.csr"
            )
            temp = io.read_bin(file_path)
            key = cp2k_settings["KS_list"][i]
            print("Reading Hamiltonian ", key)
            if temp.size != 0:
                temp = io.bin_to_sparse(temp) * HARTREE_TO_EV
                print(temp.shape)
                if key == (0, 0, 0):
                    size_mat = temp.shape[0]
                temp = temp.tocoo()
                temp._shape = (size_mat, size_mat)  # Ensure the shape is set correctly
                temp = prune_small(temp.tocsr(), eps)
                if temp.nnz > 0:
                    print(
                        "Hamiltonian Is not empty, I'm saving it. Size for",
                        key,
                        " is: ",
                        temp.size,
                    )
                    self.H[key] = temp

        # Load the Overlap for all the k-points
        for i in range(cp2k_settings["n_S"]):
            file_path = (
                path
                + "/"
                + cp2k_settings["project_name"]
                + "-S_SPIN_1_R_"
                + str(i + 1)
                + "-1_0.csr"
            )
            temp = io.read_bin(file_path)
            key = cp2k_settings["S_list"][i]
            print("Reading Overlap ", key)
            if temp.size != 0:
                temp = io.bin_to_sparse(temp)
                temp = temp.tocoo()
                temp._shape = (size_mat, size_mat)  # Ensure the shape is set correctly
                temp = prune_small(temp.tocsr(), eps * 0.1)
                if temp.nnz > 0:
                    print(
                        "Overlap Is not empty, I'm saving it. Size for",
                        key,
                        " is: ",
                        temp.size,
                    )
                    self.S[key] = temp

        # All couplings are available
        self.coup_available = [1, 1, 1]
        self.bs_available = True

    def load_from_MALOQ_GAMMA(
        self,
        H_file=None,
        S_file=None,
        xyz_file=None,
        orb_map=None,
        extract_x: bool = True,
        extract_y: bool = False,
        extract_z: bool = False,
        symmetrize: bool = True,
        eps: float = 1e-5,
        wrap: bool = True,
    ):
        """
        Load Hamiltonian and Overlap matrices from CP2K output files at the Gamma point,
        and extract coupling matrices in specified directions.

        Parameters:
        -----------
        H_file : str
            Path to the Hamiltonian matrix file.
        S_file : str
            Path to the Overlap matrix file.
        xyz_file : str
            Path to the XYZ file containing atomic coordinates.
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

        self._flag_extracted_directions(extract_x, extract_y, extract_z)

        # Extract the orb map
        if orb_map is None:
            raise ValueError("orb_map must be provided")
        self.orb_map = orb_map

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = io.read_xyz(
            xyz_file
        )

        if wrap:
            # Wrap the coordinates in the unit cell
            self.lattice["coords"] = (
                self.lattice["coords"] @ np.linalg.inv(self.lattice["L"]) % 1.0
            ) @ self.lattice["L"]

        # Extract the Fermi level
        self.Fermi = -999
        # Extract the starting orbital index for each atom
        self._refresh_orb_per_at()

        # Read the KS file (GAMMA)
        if H_file.endswith(".npy"):
            KS = np.load(H_file)
            KS = sp.csr_matrix(KS) * HARTREE_TO_EV
        elif H_file.endswith(".npz"):
            KS = sp.load_npz(H_file) * HARTREE_TO_EV
        else:
            raise ValueError("H_file must be a .npy or .npz file")

        if S_file is None:
            S = sp.eye(KS.shape[0], format="csr")
        else:
            # Read the S file (GAMMA)
            if S_file.endswith(".npy"):
                S = np.load(S_file)
                S = sp.csr_matrix(S)
            elif S_file.endswith(".npz"):
                S = sp.load_npz(S_file)
            else:
                raise ValueError("S_file must be a .npy or .npz file")

        self._store_gamma_blocks(KS, S, symmetrize, eps)

        # Extract the coupling matrices in the specified directions
        self._extract_requested_couplings(extract_x, extract_y, extract_z)

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

        self._flag_extracted_directions(extract_x, extract_y, extract_z)

        # Read the CP2K settings from the output file
        cp2k_settings = io.read_cp2k_file(path + "/" + filename)

        # Extract the orb map
        self.orb_map = cp2k_settings["no_orb"]

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = io.read_xyz(
            path + "/" + cp2k_settings["coordFile"]
        )

        # Extract the Fermi level
        self.Fermi = cp2k_settings["fermi"]
        # Extract the starting orbital index for each atom
        self._refresh_orb_per_at()

        # Read the KS file (GAMMA)
        KS = (
            io.bin_to_sparse(io.read_bin(path + "/" + cp2k_settings["KSfile"]))
            * HARTREE_TO_EV
        )
        S = io.bin_to_sparse(io.read_bin(path + "/" + cp2k_settings["Sfile"]))

        self._store_gamma_blocks(KS, S, symmetrize, eps)

        # Extract the coupling matrices in the specified directions
        self._extract_requested_couplings(extract_x, extract_y, extract_z)

    def _flag_extracted_directions(
        self, extract_x: bool, extract_y: bool, extract_z: bool
    ):
        """
        Mark the directions whose couplings are about to be extracted as
        available, so that the band structure can be computed.
        """
        for dir, extract in enumerate((extract_x, extract_y, extract_z)):
            if extract:
                self.coup_available[dir] = 1
                self.bs_available = True

    def _store_gamma_blocks(
        self, KS: sp.spmatrix, S: sp.spmatrix, symmetrize: bool, eps: float
    ):
        """
        Symmetrize, threshold and store the Gamma-point Hamiltonian and Overlap.

        Parameters:
        -----------
        KS : sp.spmatrix
            Gamma-point Hamiltonian, already converted to eV.
        S : sp.spmatrix
            Gamma-point Overlap.
        symmetrize : bool
            Whether to symmetrize the matrices.
        eps : float
            Threshold below which matrix elements are set to zero. The overlap
            uses a threshold ten times smaller.
        """
        # Symmetrize the matrices
        if symmetrize:
            KS = symmetrize_block(KS)
            S = symmetrize_block(S)

        # Remove elements below the threshold
        prune_small(KS, eps)
        prune_small(S, eps * 0.1)

        # Save the Gamma matrices
        self.H[(0, 0, 0)] = KS.tocsr()
        self.S[(0, 0, 0)] = S.tocsr()

    def _extract_requested_couplings(
        self, extract_x: bool, extract_y: bool, extract_z: bool
    ):
        """
        Extract, from the Gamma-point matrices, the coupling blocks along every
        requested direction.
        """
        for dir, extract in enumerate((extract_x, extract_y, extract_z)):
            if extract:
                self._extract_coup_from_gamma(self.H, dir)
                self._extract_coup_from_gamma(self.S, dir)

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
        sorted_orb = get_orb_from_at(sorted_at, self.start_orb_per_at)[None, :]
        sorted_orb_back = np.argsort(sorted_orb.squeeze())[None, :]

        for key in list(M.keys()):
            # Sort the hamiltonian and overlap matrix
            M_S = M[key][sorted_orb.T, sorted_orb]

            if key[dir] != 0:
                raise Exception(
                    "The hamiltonian has already not Gamma couplings in the selected direction"
                )

            # Extract the three matrices (cut over the shifted diagonal)
            bnd = int(np.round(np.shape(M_S)[0] / 2)) + 1

            M_3 = get_upper_block(M_S, bnd)

            # Check that there is a clear separation between the in-cell and coupling blocks
            check = sp.tril(M_3, bnd + 20).nnz
            if check != 0:
                print(
                    "WARNING! There is not a clear separation between the in-cell and coupling blocks. Try increasing the size of the unit cell in the direction ",
                    dir,
                )

            M_5 = get_lower_block(M_S, -bnd)
            check = sp.triu(M_5, -bnd - 20).nnz
            if check != 0:
                print(
                    "WARNING! There is not a clear separation between the in-cell and coupling blocks. Try increasing the size of the unit cell in the direction ",
                    dir,
                )

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
