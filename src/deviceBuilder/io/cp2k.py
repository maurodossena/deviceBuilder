"""Readers for CP2K output files and the binary CSR matrices they produce."""

import os
import struct

import numpy as np
from scipy import sparse

from ..constants import HARTREE_TO_EV_OUTPUT


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
                    cp2k_settings["lowMO"] = float(line.split()[0]) * HARTREE_TO_EV_OUTPUT
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
                cp2k_settings["fermi"] = round(
                    float(line.split()[5]) * HARTREE_TO_EV_OUTPUT, 2
                )
            if "Fermi Energy" in line:
                cp2k_settings["fermi"] = round(float(line.split()[4]), 2)
            if "Fermi energy:" in line:
                cp2k_settings["fermi"] = round(
                    float(line.split()[2]) * HARTREE_TO_EV_OUTPUT, 2
                )
            if "MO| E(Fermi):" in line:
                cp2k_settings["fermi"] = round(float(line.split()[4]), 2)
            if "Eigenvalues of the occupied subspace spin            1" in line:
                lineCount = 2

    if "n_KS" in cp2k_settings:
        cp2k_settings["KS_list"] = _read_cell_list(filename, "KS CSR write|", cp2k_settings["n_KS"])

    if "n_S" in cp2k_settings:
        cp2k_settings["S_list"] = _read_cell_list(filename, "S CSR write|", cp2k_settings["n_S"])

    return cp2k_settings


def _read_cell_list(filename: str, marker: str, n_cells: int) -> list:
    """
    Read the list of cell indices printed after a `CSR write` marker.

    Parameters
    ----------
    filename : str
        CP2K output file to be read.
    marker : str
        Line marking the beginning of the block ("KS CSR write|" or "S CSR write|").
    n_cells : int
        Number of cells listed in the block.
    Returns
    -------
    list
        List of (A, B, C) cell indices.
    """
    cell_list = []
    with open(filename, "r") as filehandle:

        while True:
            line = filehandle.readline()
            if marker in line:
                line = filehandle.readline()
                for i in range(n_cells):
                    line = filehandle.readline()
                    A = int(line.split()[1])
                    B = int(line.split()[2])
                    C = int(line.split()[3])
                    cell_list.append((A, B, C))

                break

    return cell_list


def read_bin(fname, struct_fmt="<IIIdI"):
    """
    Parameters
    ----------
    fname : File to be read as '*.csr'
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
    -------
    N : A Scipy sparse matrix (*x*) containing the orbital coupling

    """
    xmax, ymax, zmax = M.max(axis=0)

    element = M[:, 2]

    x_ind = M.astype(int)[:, 0] - 1
    y_ind = M.astype(int)[:, 1] - 1

    N = sparse.csr_matrix((element, (x_ind, y_ind)), shape=(int(xmax), int(ymax)))

    return N
