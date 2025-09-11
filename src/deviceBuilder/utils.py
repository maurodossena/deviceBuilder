import os
import struct

import numpy as np
from scipy import sparse

factor = 27.2114

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

def read_xyz(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an xyz file and returns the lattice(3x3), atoms(N) and coordinates(Nx3)
    Parameters
    ----------
    filename : str
        File to be read as '*.xyz'
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
    lattice = np.asarray(lattice, dtype=np.float64)

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
