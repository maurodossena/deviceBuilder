"""Loading a device from CP2K, Wannier90 or MALOQ output."""

import numpy as np
from scipy import sparse as sp

from . import io
from .base import DeviceBase
from .constants import HARTREE_TO_EV
from .lattice import get_orb_from_at
from .sparse_ops import get_lower_block, get_upper_block, prune_small

EPS_HERMITIAN = 1e-10

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
        w_centers, self.lattice["L"], lattice_coods, lattice_at = (
            io.read_wannier_wout(
                path + "/" + filename, return_atom=True
            )
        )

        self.orb_map["X"] = 1
        
        self.lattice["coords"] = w_centers.copy()
        self.lattice["at"] = np.array(["X"] * len(w_centers))

        h_wann = io.read_hr_dat(path + "/" + filename.replace(".wout", "_hr.dat"))

        n_coup_1 = h_wann.shape[0]
        n_coup_2 = h_wann.shape[1]
        n_coup_3 = h_wann.shape[2]

        for i in range(-(n_coup_1 // 2), (n_coup_1 // 2) + 1):
            for j in range(-(n_coup_2 // 2), (n_coup_2 // 2) + 1):
                for k in range(-(n_coup_3 // 2), (n_coup_3 // 2) + 1):
                    temp = h_wann[i, j, k, :, :]
                    temp = prune_small(sp.csr_matrix(temp), eps)
                    if temp.size != 0:
                        self.H[(i, j, k)] = temp

        self.S[(0, 0, 0)] = sp.eye(self.H[(0, 0, 0)].shape[0], format="csr")

        self._refresh_orb_per_at()

        print("WARNING! The Fermi level is not available in the Wannier90 output. It is set to -999 eV. Please set it manually if needed.")
        self.Fermi = -999
        
        self._symmetrize_if_upper_triangular("H")

        self._update_max_coupling()

        self._add_missing_couplings(self.H[(0,0,0)].shape[0], "H")
        self._add_missing_couplings(self.S[(0,0,0)].shape[0], "S")

        self._check_if_hermitian("H")
        self._check_if_hermitian("S")

        self.PBC = [True, True, True]  # Wannier90 output is always periodic in all directions

        if gather_wannier_centers:

            self.wrap_cell()

            w_centers = self.lattice["coords"].copy()

            #wrap lattice coordinates to the unit cell
            self.lattice["coords"] = np.mod(lattice_coods @ np.linalg.inv(self.lattice["L"]), 1) @ self.lattice["L"]
            self.lattice["at"]  = lattice_at
            
            elements = np.unique(lattice_at)

            for at in elements:
                self.orb_map[at] = 0

            del self.orb_map["X"]

            grid = np.array(
                np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
            )
            combos = grid.reshape(3, -1).T
            translations = combos @ self.lattice["L"]

            translated_atoms = (
                self.lattice["coords"][:,:, None] + translations.T[None, :, :]
            )

            w_atoms = np.zeros(len(w_centers), dtype=int)
            for i, w in enumerate(w_centers):
                dist = np.min(
                    np.linalg.norm(w[None, :, None] - translated_atoms, axis=1), axis=1
                )
                index = np.argmin(dist)
                w_atoms[i] = index

            sorted_index = np.argsort(w_atoms)
            unique_at, orb_counts = np.unique(w_atoms, return_counts=True)

            for i in range(len(unique_at)):
                at_i = self.lattice["at"][unique_at[i]]
                if self.orb_map[at_i] == 0:
                    self.orb_map[at_i] = orb_counts[i]
                else:
                    if self.orb_map[at_i] != orb_counts[i]:
                        raise Exception(
                            "Error in gathering the Wannier centers: different number of orbitals found for atom ",
                            at_i,
                        )

            for n in self.orb_map:
                if self.orb_map[n] == 0:
                    raise Exception(
                        "Error in gathering the Wannier centers: no orbitals found for atom ",
                        n,
                    )

            for key in self.H.keys():
                self.H[key] = self.H[key][sorted_index, :][:, sorted_index]

            self._refresh_orb_per_at()

            self._check_if_hermitian("H")
            self._check_if_hermitian("S")

            self._update_max_coupling()
            
            self._add_missing_couplings(self.H[(0,0,0)].shape[0], "H")
            self._add_missing_couplings(self.S[(0,0,0)].shape[0], "S")


    def load_from_MALOQ_GAMMA(
        self,
        H_file=None,
        S_file=None,
        xyz_file=None,
        orb_map=None,
        eps: float = 1e-5,
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

        # Extract the orb map
        if orb_map is None:
            raise ValueError("orb_map must be provided")
        self.orb_map = orb_map

        # Extract the lattice infos. L is a 3x3 numpy matrix with the lattice vectors as rows, at is a numpy array with the atom types, coords is a Nx3 numpy array with the atomic coordinates
        self.lattice = {}
        self.lattice["L"], self.lattice["at"], self.lattice["coords"] = io.read_xyz(
            xyz_file
        )

        # Extract the Fermi level
        self.Fermi = 0
        # Extract the starting orbital index for each atom
        self._refresh_orb_per_at()

        # Read the KS file (GAMMA)
        if H_file.endswith(".npy"):
            self.H[(0,0,0)] = np.load(H_file)
            self.H[(0,0,0)] = sp.csr_matrix(self.H[(0,0,0)]) * HARTREE_TO_EV
        elif H_file.endswith(".npz"):
            self.H[(0,0,0)] = sp.load_npz(H_file) * HARTREE_TO_EV
        else:
            raise ValueError("H_file must be a .npy or .npz file")

        if S_file is None:
            self.S[(0,0,0)] = sp.eye(self.H[(0,0,0)].shape[0], format="csr")
        else:
            # Read the S file (GAMMA)
            if S_file.endswith(".npy"):
                self.S[(0,0,0)] = np.load(S_file)
                self.S[(0,0,0)] = sp.csr_matrix(self.S[(0,0,0)])
            elif S_file.endswith(".npz"):
                self.S[(0,0,0)] = sp.load_npz(S_file)
            else:
                raise ValueError("S_file must be a .npy or .npz file")

        self._prune_matrices(eps)
        
        self._symmetrize_if_upper_triangular("H")
        self._symmetrize_if_upper_triangular("S")

        self._extract_couplings_from_gamma(0)
        self._extract_couplings_from_gamma(1)
        self._extract_couplings_from_gamma(2)

        self._update_max_coupling()

        self._add_missing_couplings(self.H[(0,0,0)].shape[0], "H")
        self._add_missing_couplings(self.S[(0,0,0)].shape[0], "S")

        self._check_if_hermitian("H")
        self._check_if_hermitian("S")

        self.PBC = [True, True, True]  # MALOQ output is always periodic in all directions

    def load_from_cp2k(
        self,
        filename: str,
        path: str = "./",
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

        # Read the CP2K settings from the output file
        cp2k_settings, self.lattice = io.read_cp2k_file(path + "/" + filename)

        # Extract the orb map
        self.orb_map = cp2k_settings["no_orb"]

        # Extract the Fermi level
        self.Fermi = cp2k_settings["fermi"]
        # Extract the starting orbital index for each atom
        self._refresh_orb_per_at()

        if not cp2k_settings["k_points"]:
            # Read the KS file (GAMMA)
            self.H[(0, 0, 0)] = io.read_csr(path + "/" + cp2k_settings["KSfile"], n=cp2k_settings["H_size"])
            self.H[(0, 0, 0)].data *= HARTREE_TO_EV
            self.S[(0, 0, 0)] = io.read_csr(path + "/" + cp2k_settings["Sfile"], n=cp2k_settings["H_size"])

        else:
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
                H_k = io.read_csr(file_path, n=cp2k_settings["H_size"])
                key = cp2k_settings["KS_list"][i]
                if H_k.size != 0:
                    H_k.data *= HARTREE_TO_EV
                    self.H[key] = H_k

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
                S_k = io.read_csr(file_path, n=cp2k_settings["H_size"])
                key = cp2k_settings["S_list"][i]
                if S_k.size != 0:
                    self.S[key] = S_k

        self._prune_matrices(eps)

        self._symmetrize_if_upper_triangular("H")
        self._symmetrize_if_upper_triangular("S")

        if not cp2k_settings["k_points"]:
            # Extract the coupling matrices in the specified directions
            self._extract_couplings_from_gamma(0)
            self._extract_couplings_from_gamma(1)
            self._extract_couplings_from_gamma(2)

        else:
            coords_rel = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"]) + 0.5
            if np.any(np.abs(coords_rel - np.round(coords_rel)) < 1e-6):
                raise Exception(
                    """The atomic coordinates are too close to the half-cell boundaries.
                    This can lead to incorrect wrapping of the Hamiltonian and Overlap matrices. 
                    Please adjust the atomic coordinates to be well within the unit cell.
                    """
                )
            
            translation = np.floor(coords_rel)

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

        self._add_missing_couplings(cp2k_settings["H_size"], "H")
        self._add_missing_couplings(cp2k_settings["H_size"], "S")

        self._check_if_hermitian("H")
        self._check_if_hermitian("S")

        self.PBC = [True, True, True]  # CP2K output is always periodic in all directions

    def _update_max_coupling(self):
        """
        Update the maximum coupling in each direction for the Hamiltonian and Overlap matrices.
        Returns a list of three integers representing the maximum coupling in the x, y, and z directions.
        """
        max_coupling = [0, 0, 0]

        for key in self.H.keys():
            for i in range(3):
                max_coupling[i] = max(max_coupling[i], abs(key[i]))

        for key in self.S.keys():
            for i in range(3):
                max_coupling[i] = max(max_coupling[i], abs(key[i]))

        print("The coupling range is: ", max_coupling)

        self.max_coupling = max_coupling
    
    def _add_missing_couplings(self, size: int, matrix_type: str = "H" ):
        """
        Tail with empty matrices to obtain a "square" set of couplings.

        Parameters:
        -----------
        matrix_type : str
            Type of matrix to add missing couplings for. Can be "H" for Hamiltonian
            or "S" for Overlap.
        size : int
            Size of the matrices to be added.
    
        """
        if matrix_type == "H":
            M = self.H
        elif matrix_type == "S":
            M = self.S
        else:
            raise ValueError("matrix_type must be 'H' or 'S'")

        for i in range(-self.max_coupling[0], self.max_coupling[0] + 1):
            for j in range(-self.max_coupling[1], self.max_coupling[1] + 1):
                for k in range(-self.max_coupling[2], self.max_coupling[2] + 1):
                    key = (i, j, k)
                    if key not in M:
                        M[key] = sp.csr_matrix((size, size))

    def _check_if_hermitian(self, matrix_type: str):
        """
        Check if the Hamiltonian and Overlap matrices are Hermitian.
        """

        if matrix_type == "H":
            M = self.H
        elif matrix_type == "S":
            M = self.S
        else:
            raise ValueError("matrix_type must be 'H' or 'S'")

        for key in M.keys():
            key_opposite = (-key[0], -key[1], -key[2])
            if key_opposite not in M:
                raise Exception(
                    "The "
                    + matrix_type
                    + " matrices are not Hermitian. Missing the opposite coupling for key: "
                    + str(key)
                )
            if M[key].shape != M[key_opposite].shape:
                raise Exception(
                    "The "
                    + matrix_type
                    + " matrices are not Hermitian. The shape of the opposite coupling for key: "
                    + str(key)
                    + " is different."
                )
            if (np.abs(M[key] - M[key_opposite].transpose().conj())>EPS_HERMITIAN).nnz != 0:
                raise Exception(
                    "The "
                    + matrix_type
                    + " matrices are not Hermitian. The opposite coupling for key: "
                    + str(key)
                    + " is not the conjugate transpose."
                )

    def _symmetrize_if_upper_triangular(self, matrix_type: str):
        """
        Check if the Hamiltonian and Overlap matrices are stored in an upper triangular format.
        If they are, symmetrize them by adding their transpose.
        """

        M = self.H if matrix_type == "H" else self.S

        for key in M.keys():
            if sp.tril(M[key], -1).nnz != 0:
                return # The matrix is not upper triangular, no need to symmetrize

        print("The " + matrix_type + " matrices were stored in an upper triangular format. They have been symmetrized.")
        for key in list(M.keys()):
            key_opposite = (-key[0], -key[1], -key[2])
            if key_opposite not in M:
                M[key_opposite] =  sp.triu(M[key], 1).transpose()
            else:
                M[key_opposite] += sp.triu(M[key], 1).transpose()


    def _prune_matrices(self, eps: float):
        """
        Prune the Hamiltonian and Overlap matrices by removing elements below a threshold.

        Parameters:
        -----------
        eps : float
            Threshold below which matrix elements are set to zero.
        """
        for key in list(self.H.keys()):
            prune_small(self.H[key], eps)
            if self.H[key].nnz == 0:
                del self.H[key]
        for key in list(self.S.keys()):
            prune_small(self.S[key], eps * 0.1)
            if self.S[key].nnz == 0:
                del self.S[key]

    def _extract_couplings_from_gamma(self, dir: int):
        """
        Extract the coupling matrices in the specified direction from the Gamma-point Hamiltonian.

        Parameters:
        -----------
            Direction along which to extract the coupling matrices.
        """

        # Compute the fractional coordinates of the atoms
        frac_coord = self.lattice["coords"] @ np.linalg.inv(self.lattice["L"])

        # Sort the atoms along the specified direction
        sorted_at = np.argsort(frac_coord[:, dir])

        # Get the sorted orbitals and the indices to go back to the original order
        sorted_orb = get_orb_from_at(sorted_at, self.start_orb_per_at)[None, :]
        sorted_orb_back = np.argsort(sorted_orb.squeeze())[None, :]

        coup_max_dir = 0
        for key in self.H.keys():
            coup_max_dir = max(coup_max_dir, abs(key[dir]))
        for key in self.S.keys():
            coup_max_dir = max(coup_max_dir, abs(key[dir]))

        if coup_max_dir != 0:
            raise Exception(
                """The Hamiltonian and Overlap matrices are not only defined at the 
                Gamma point. Coupling matrices in this direction are already available."""
            )

        #Run a first loop to check if coupling can be extracted
        for key in list(self.H.keys()):
            #Since we start from Gamma, we assume H and S have the same keys
            
            # Sort the hamiltonian and overlap matrix
            H_S = self.H[key][sorted_orb.T, sorted_orb]
            S_S = self.S[key][sorted_orb.T, sorted_orb]

            # Extract the three matrices (cut over the shifted diagonal)
            bnd = int(np.round(np.shape(H_S)[0] / 2)) + 1

            H_3 = get_upper_block(H_S, bnd)
            S_3 = get_upper_block(S_S, bnd)

            # Check that there is a clear separation between the in-cell and coupling blocks
            check1 = sp.tril(H_3, bnd + 50).nnz
            check2 = sp.tril(S_3, bnd + 50).nnz

            if check1 != 0 or check2 != 0:
                print(f"""For direction {dir}, there is not a clear separation between 
                      the in-cell and coupling blocks. Coupling blocks in this direction 
                      are not extracted.""")
                return
        
        for key in list(self.H.keys()):
            #Since we start from Gamma, we assume H and S have the same keys
            
            # Sort the hamiltonian and overlap matrix
            H_S = self.H[key][sorted_orb.T, sorted_orb]
            S_S = self.S[key][sorted_orb.T, sorted_orb]

            # Extract the three matrices (cut over the shifted diagonal)
            bnd = int(np.round(np.shape(H_S)[0] / 2)) + 1

            H_3 = get_upper_block(H_S, bnd)
            S_3 = get_upper_block(S_S, bnd)

            H_5 = get_lower_block(H_S, -bnd)
            S_5 = get_lower_block(S_S, -bnd)


            H_4 = H_S - H_3
            H_4 = H_4 - H_5
            H_4.eliminate_zeros()
            S_4 = S_S - S_3
            S_4 = S_4 - S_5
            S_4.eliminate_zeros()

            # Reorder the matrices to the original order
            H_3 = H_3[sorted_orb_back.T, sorted_orb_back]
            S_3 = S_3[sorted_orb_back.T, sorted_orb_back]

            H_4 = H_4[sorted_orb_back.T, sorted_orb_back]
            S_4 = S_4[sorted_orb_back.T, sorted_orb_back]

            H_5 = H_5[sorted_orb_back.T, sorted_orb_back]
            S_5 = S_5[sorted_orb_back.T, sorted_orb_back]

            key_l = list(key)

            # Update the dictionaries
            self.H[key] = H_4
            self.S[key] = S_4

            key_l[dir] = 1
            self.H[tuple(key_l)] = H_5
            self.S[tuple(key_l)] = S_5

            key_l[dir] = -1
            self.H[tuple(key_l)] = H_3
            self.S[tuple(key_l)] = S_3
