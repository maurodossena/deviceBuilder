import os
import struct

from pathlib import Path
import numpy as np
from scipy import sparse

import re

factor = 27.2114


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

    TODO: Add tests.

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


def print_bin(filename, M):
    """
    Save the binary file containing the matrix M

    Parameters
    ----------
    filename : Name of the bin file

    M : Matrix to be put in the matrix

    eps : treshold for value to be set at 0

    Returns
    -------
    None.

    """

    # Build a csr rapresentation
    [indx, indy, val] = sparse.find(M)
    indices = np.array([indx, indy])
    values = np.array(val)

    M_4 = np.column_stack((np.transpose(indices), values))
    M_4 = M_4 + [1, 1, 0]
    M_4 = np.column_stack(
        (M_4[:, 0], M_4[:, 1], np.real(M_4[:, 2]), np.imag(M_4[:, 2]))
    )
    header = [np.shape(M)[0], np.shape(M_4)[0], 1]

    index = np.lexsort((M_4[:, 1], M_4[:, 0]))
    M_4 = M_4[index]

    # write the bin file
    M_4_write = np.reshape(M_4, (M_4.size, 1))
    np.concatenate((np.reshape(header, (3, 1)), M_4_write)).astype("double").tofile(
        os.path.join("./", filename)
    )


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


def read_xyz(filename: str, decimals=6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an xyz file and returns the lattice(3x3), atoms(N) and coordinates(Nx3)
    Parameters
    ----------
    filename : str
        File to be read as '*.xyz'
    decimals : int
        Number of decimal places to round the coordinates.
    Returns
    -------
    lattice : np.ndarray
        Lattice vectors of the system.
    atoms : np.ndarray
        Atom types of the system.
    coords : np.ndarray
        Coordinates of the atoms.
    """

    atoms = []
    coords = []
    lattice = []

    with open(filename, "rt") as myfile:
        for line in myfile:
            # num_atoms line
            if len(line.split()) == 1:
                pass
            # blank line
            elif len(line.split()) == 0:
                pass
            # line with cell parameters
            elif "Lattice=" in line:
                lattice = line.replace('Lattice="', "")
                lattice = lattice.replace('"', "")
                lattice = np.reshape(lattice.split()[0:9], (3, -1))

            # line with atoms and positions
            elif len(line.split()) == 4:
                c = line.split()[0]
                atoms.append(c)
                coords.append(line.split()[1:])
            else:
                pass

    atoms = np.asarray(atoms, dtype=str)
    coords = np.asarray(coords, dtype=np.float64)
    coords = np.round(coords, decimals=decimals)
    lattice = np.asarray(lattice, dtype=np.float64)
    lattice = np.round(lattice, decimals=decimals)

    return lattice, atoms, coords


def read_cp2k_file(filename: str) -> dict:
    """
    Reads a CP2K output file and returns a dictionary with the relevant information
    Parameters
    ----------
    filename : str
        File to be read as CP2K output file
    Returns
    -------
    cp2k_settings : dict
        Dictionary containing the relevant information from the CP2K output file
    """

    cp2k_settings = {}
    cp2k_settings["no_orb"] = {}

    index = 0
    lineCount = 0
    with open(filename, "r") as filehandle:
        for line in filehandle:

            if lineCount > 0:
                if lineCount == 1:
                    cp2k_settings["lowMO"] = float(line.split()[0]) * factor
                lineCount -= 1

            if "Atomic kind" in line:
                last_at = line.split()[3]
            if "Number of spherical" in line:
                cp2k_settings["no_orb"][last_at] = int(line.split()[5])
                index = index + 1
            if "Project name" in line:
                cp2k_settings["KSfile"] = line.split()[3] + "-KS_SPIN_1-1_0.csr"
                cp2k_settings["Sfile"] = line.split()[3] + "-S_SPIN_1-1_0.csr"
                cp2k_settings["project_name"] = line.split()[3]
            if "KS CSR write|" in line:
                cp2k_settings["n_KS"] = int(line.split()[3])
            if "S CSR write|" in line:
                cp2k_settings["n_S"] = int(line.split()[3])
            if "Coordinate file name" in line:
                cp2k_settings["coordFile"] = line.split()[4]
            if "Fermi level:" in line:
                cp2k_settings["fermi"] = round(float(line.split()[5]) * factor, 2)
            if "Fermi Energy" in line:
                cp2k_settings["fermi"] = round(float(line.split()[4]), 2)
            if "Fermi energy:" in line:
                cp2k_settings["fermi"] = round(float(line.split()[2]) * factor, 2)
            if "MO| E(Fermi):" in line:
                cp2k_settings["fermi"] = round(float(line.split()[4]), 2)
            if "Eigenvalues of the occupied subspace spin            1" in line:
                lineCount = 2

    if "n_KS" in cp2k_settings:
        cp2k_settings["KS_list"] = []
        with open(filename, "r") as filehandle:

            while True:
                line = filehandle.readline()
                if "KS CSR write|" in line:
                    line = filehandle.readline()
                    for i in range(cp2k_settings["n_KS"]):
                        line = filehandle.readline()
                        A = int(line.split()[1])
                        B = int(line.split()[2])
                        C = int(line.split()[3])
                        cp2k_settings["KS_list"].append((A, B, C))

                    break

    if "n_S" in cp2k_settings:
        cp2k_settings["S_list"] = []
        with open(filename, "r") as filehandle:

            while True:
                line = filehandle.readline()
                if "S CSR write|" in line:
                    line = filehandle.readline()
                    for i in range(cp2k_settings["n_S"]):
                        line = filehandle.readline()
                        A = int(line.split()[1])
                        B = int(line.split()[2])
                        C = int(line.split()[3])
                        cp2k_settings["S_list"].append((A, B, C))

                    break

    return cp2k_settings


def read_bin(fname, struct_fmt="<IIIdI"):
    """
    Parameters
    ----------
    fname : File to be read as '*.xyz'
    struct_fmt : Structure of the binary file. The default is "<IIIdI".

    Returns
    -------
    M : A (*x3) Numpy array with cell indexes and orbital coupling

    """

    # Obtain the struct length
    struct_len_I = struct.calcsize(struct_fmt)
    struct_unpack_I = struct.Struct(struct_fmt).unpack_from

    # Obtain the file length and inizialize matrix
    filesize = os.path.getsize(fname)
    num_lines = int(filesize / struct_len_I)
    M = np.zeros((num_lines, 3))
    print("Loading " + fname + "...")

    # Load matrix
    with open(fname, "rb") as f:
        for ind in range(num_lines):
            data = f.read(struct_len_I)
            # Ignore the headers at the start and end of a block
            M[ind] = struct_unpack_I(data)[1:-1]

    return M


def bin_to_sparse(M):
    """
    Convert the pseudo-csr matrix in a true sparse csr matrix

    Parameters
    ----------
    M : A (*x3) Numpy array containing atomic indexes and the orbital coupling

    Returns
    -------ASE provides a module, ase.cluster, to set up metal nanoparticles with common crystal forms. Please have a quick look at the documentation.
    N : A Scipy sparse matrix (*x*) containing the orbital coupling

    """
    xmax, ymax, zmax = M.max(axis=0)

    element = M[:, 2]

    x_ind = M.astype(int)[:, 0] - 1
    y_ind = M.astype(int)[:, 1] - 1

    N = sparse.csr_matrix((element, (x_ind, y_ind)), shape=(int(xmax), int(ymax)))

    return N


def symmetrize_block(block: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Symmetrize a sparse matrix.
    Parameters
    ----------
    block : sparse.csr_matrix
        Matrix to be symmetrized.
    Returns
    -------
    sparse.csr_matrix
        Symmetrized matrix.
    """
    upper = sparse.triu(block, 1)
    return block + upper.T


def get_upper_block(block: sparse.csr_matrix, cutoff_diag: int) -> sparse.csr_matrix:
    """
    Get the upper block of a sparse matrix.
    Parameters
    ----------
    block : sparse.csr_matrix
        Matrix from which to extract the upper block.
    cutoff_diag : int
        Diagonal cutoff for the upper block.
    Returns
    -------
    sparse.csr_matrix
        Upper block of the matrix.
    """
    upper = sparse.triu(block, cutoff_diag, format="csr")
    return upper


def get_lower_block(block: sparse.csr_matrix, cutoff_diag: int) -> sparse.csr_matrix:
    """
    Get the lower block of a sparse matrix.
    Parameters
    ----------
    block : sparse.csr_matrix
        Matrix from which to extract the lower block.
    cutoff_diag : int
        Diagonal cutoff for the lower block.
    Returns
    -------
    sparse.csr_matrix
        Lower block of the matrix.
    """
    lower = sparse.tril(block, cutoff_diag, format="csr")
    return lower
