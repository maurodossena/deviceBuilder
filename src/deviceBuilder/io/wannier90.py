"""Readers for Wannier90 output files."""

import re
from pathlib import Path

import numpy as np


def read_hr_dat(
    path: Path, return_all: bool = False, dtype=np.complex128, read_fast=False
):
    """Parses the contents of a `seedname_hr.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    `num_wann`. The third line gives the number of Wigner-Seitz
    grid-points.

    The next block of integers gives the degeneracy of each Wigner-Seitz
    grid point, arranged into 15 values per line.

    Finally, the remaining lines each contain, respectively, the
    components of the Wigner-Seitz cell index, the Wannier center
    indices m and n, and and the real and imaginary parts of the
    Hamiltonian matrix element `HRmn` in the localized basis.

    Parameters
    ----------
    path : Path
        Path to a `seedname_hr.dat` file.
    return_all : bool, optional
        Whether to return all the data or just the Hamiltonian in the
        localized basis. When `True`, the degeneracies and the
        Wigner-Seitz cell indices are also returned. Defaults to
        `False`.
    dtype : dtype, optional
        The data type of the Hamiltonian matrix elements. Defaults to
        `numpy.complex128`.
    read_fast : bool, optional
        Whether to assume that the file is well-formatted and all the
        data is sorted correctly. Defaults to `False`.

    Returns
    -------
    hr : ndarray
        The Hamiltonian matrix elements in the localized basis.
    degeneracies : ndarray, optional
        The degeneracies of the Wigner-Seitz grid points.
    R : ndarray, optional
        The Wigner-Seitz cell indices.

    """

    # Strip info from header.
    num_wann, nrpts = np.loadtxt(path, skiprows=1, max_rows=2, dtype=int)
    num_wann, nrpts = int(num_wann), int(nrpts)

    # Read wannier data (skipping degeneracy info).
    deg_rows = int(np.ceil(nrpts / 15.0))
    wann_dat = np.loadtxt(path, skiprows=3 + deg_rows)

    # Assign R
    if read_fast:
        R = wann_dat[:: num_wann**2, :3].astype(int)
    else:
        R = wann_dat[:, :3].astype(int)
    Rs = np.subtract(R, R.min(axis=0))
    N1, N2, N3 = Rs.max(axis=0) + 1
    N1, N2, N3 = int(N1), int(N2), int(N3)

    # Obtain Hamiltonian elements.
    if read_fast:
        hR = wann_dat[:, 5] + 1j * wann_dat[:, 6]
        hR = hR.reshape(N1, N2, N3, num_wann, num_wann).swapaxes(-2, -1)
        hR = np.roll(hR, shift=(N1 // 2 + 1, N2 // 2 + 1, N3 // 2 + 1), axis=(0, 1, 2))
    else:
        hR = np.zeros((N1, N2, N3, num_wann, num_wann), dtype=dtype)
        for line in wann_dat:
            R1, R2, R3 = line[:3].astype(int)
            m, n = line[3:5].astype(int)
            hR_mn_real, hR_mn_imag = line[5:]
            hR[R1, R2, R3, m - 1, n - 1] = hR_mn_real + 1j * hR_mn_imag

    if return_all:
        return hR, np.unique(R, axis=0)
    return hR


def read_wannier_wout(
    path: Path, transform_home_cell: bool = True, return_atom: bool = False
):
    """Parses the contents of a `seedname.wout` file and returns the Wannier centers and lattice vectors.

    Parameters
    ----------
    path : Path
        Path to a `seedname.wout` file.
    transform_home_cell : bool, optional
        Whether to transform the Wannier centers to the home cell. Defaults to `True`.
    return_atom : bool, optional
        Whether to return the atomic coordinates and elements. Defaults to `False`.

    Returns
    -------
    wannier_centers : ndarray
        The Wannier centers.
    lattice_vectors : ndarray
        The lattice vectors.
    atom_coords : ndarray, optional
        The atomic coordinates.
    atom_elements : ndarray, optional
        The atomic elements.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)

    # Find the line with the lattice vectors.
    for i, line in enumerate(lines):
        if "Lattice Vectors" in line:
            lattice_vectors = np.asarray(
                [list(map(float, lines[i + j + 1].split()[1:])) for j in range(3)]
            )
        if "Number of Wannier Functions" in line:
            num_wann = int(line.split()[-2])
            break

    # Find the line with the Wannier centers. Start from the end of the file.
    for i, line in enumerate(lines[::-1]):
        if "Final State" in line:
            # The Wannier centers are enclosed by parantheses, so we have to extract them.
            wannier_centers = np.asarray(
                [
                    list(
                        map(
                            float,
                            re.findall(r"\((.*?)\)", lines[num_lines - i + j])[0].split(
                                ","
                            ),
                        )
                    )
                    for j in range(num_wann)
                ]
            )
            break

    if transform_home_cell:
        # Get the transformation that diagonalize the lattice vectors
        transformation = np.linalg.inv(lattice_vectors)
        # Appy it to the wannier centers
        wannier_centers = np.dot(wannier_centers, transformation)
        # Translate the Wannier centers to the home cell
        wannier_centers = np.mod(wannier_centers, 1)
        # Transform the Wannier centers back to the original basis
        wannier_centers = np.dot(wannier_centers, lattice_vectors)

    if not return_atom:
        return wannier_centers, lattice_vectors

    # Extract atomic coordinates and elements
    # Regex pattern to match lines with atomic coordinates
    coord_pattern = re.compile(
        r"\|\s+(\w+)\s+(\d+)\s+[\d\.\s-]+\|\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    )

    atom_elements = []
    atom_coords = []

    # Find the section with atomic coordinates
    for i in range(len(lines)):
        if "Site" in lines[i]:
            j = i + 2
            while True:
                match = coord_pattern.search(lines[j])
                # Now we loop until we find no more matches
                if match:
                    atom_elements.append(match.group(1))
                    atom_coords.append(
                        np.asarray([float(match.group(k)) for k in range(3, 6)])
                    )
                    j += 1
                else:
                    break
            break

    # Convert to NDArrays
    atom_coords = np.array(atom_coords)
    atom_elements = np.array(atom_elements)

    if transform_home_cell:
        atom_coords = np.dot(atom_coords, transformation)
        atom_coords = np.mod(atom_coords, 1)
        atom_coords = np.dot(atom_coords, lattice_vectors)

    return wannier_centers, lattice_vectors, atom_coords, atom_elements
