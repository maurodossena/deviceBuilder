"""Band structure of a periodic device."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp

from .base import DeviceBase

try:  # GPU acceleration is optional
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


class BandStructureMixin(DeviceBase):
    """Diagonalization of H(k) along a path in reciprocal space."""

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
        cond_min : float
            Minimum conduction band energy.
        val_max : float
            Maximum valence band energy.
        """

        if not self.bs_available:
            raise Exception(
                "I deleted some elements of the hamiltonian, I can't compute the band structure"
            )

        k_points = self._build_k_points(k_path, n_point)
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

            S_k = S_k.todense()

            if CUPY_AVAILABLE:
                # Use CuPy for GPU acceleration
                S_chol_k = cp.linalg.cholesky(cp.asarray(S_k))
                del S_k  # Free GPU memory
                S_inv_k = cp.linalg.inv(S_chol_k)
                H_k = H_k.todense()
                H_ortho_k = S_inv_k @ cp.asarray(H_k) @ S_inv_k.T.conj()
                del S_chol_k, S_inv_k  # Free GPU memory
                eigvals = cp.linalg.eigvalsh(H_ortho_k)

                eigvals = eigvals.get()  # Transfer back to CPU

            else:
                S_chol_k = np.linalg.cholesky(S_k)
                S_inv_k = np.linalg.inv(S_chol_k)
                H_ortho_k = S_inv_k @ H_k @ S_inv_k.T.conj()

                # Compute the eigenvalues around the Fermi level
                eigvals = np.linalg.eigvalsh(H_ortho_k)

            # Plot the eigenvalues at that k-point
            plt.scatter([i] * len(eigvals), eigvals, color="black", s=1)

            # Append the eigenvalues to the total list
            eig_tot.append(eigvals)

        plt.ylim(self.Fermi - dE, self.Fermi + dE)
        plt.show()

        eig_tot = np.column_stack(eig_tot)

        eig_temp = eig_tot.copy()
        eig_temp[eig_temp < self.Fermi] = 1e10

        cond_min = np.min(eig_temp)
        print("Conduction band minimum: ", np.min(eig_temp))

        eig_temp = eig_tot.copy()
        eig_temp[eig_temp > self.Fermi] = -1e10

        val_max = np.max(eig_temp)
        print("Valence band maximum: ", np.max(eig_temp))

        return eig_tot, cond_min, val_max

    def _build_k_points(self, k_path: list, n_point: int):
        """
        Interpolate `n_point` k-points between every pair of vertices of `k_path`.

        Parameters:
        -----------
        k_path : list
            List of k-points defining the path in reciprocal space.
        n_point : int
            Number of points to interpolate between each pair of k-points.
        Returns:
        --------
        np.ndarray
            The k-points along the path.
        """

        k_points = []
        for i in range(len(k_path) - 1):
            k_start = np.array(k_path[i])
            k_end = np.array(k_path[i + 1])

            for dir, name in enumerate("xyz"):
                if k_start[dir] != 0 or k_end[dir] != 0:
                    if self.coup_available[dir] == 0:
                        raise Exception(
                            f"I don't have the coupling in {name} direction"
                        )

            segment = [
                k_start + (k_end - k_start) * t / n_point for t in range(n_point)
            ]
            k_points.extend(segment)
        k_points.append(k_path[-1])

        return np.array(k_points)
