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
        Dictionary containing the relevant information from the CP2K output file.
    lattice : dict
        Dictionary containing the lattice information from the CP2K output file.
    """

    cp2k_settings = {}
    cp2k_settings["no_orb"] = {}
    cp2k_settings["k_points"] = False
    lattice = {}

    index = 0
    with open(filename, "r") as filehandle:
        for line in filehandle:
            if line.split()[:2] == ["CELL|", "Vector"]:
                # " CELL| Vector a [angstrom]:  x  y  z   |a| =  norm"
                # The components are printed with 3 decimals only, while the
                # norm has 6: rescale the vector to recover the precision.
                fields = line.split()
                vec = np.asarray(fields[4:7], dtype=np.float64)
                norm = float(fields[9])
                lattice[fields[2]] = vec / np.linalg.norm(vec) * norm
            if "Atomic kind" in line:
                last_at = line.split()[3]
            if "- Spherical basis functions:" in line:
                cp2k_settings["H_size"] = int(line.split()[4])
            if "Number of spherical" in line:
                cp2k_settings["no_orb"][last_at] = int(line.split()[5])
                index = index + 1
            if "Project name" in line:
                cp2k_settings["KSfile"] = line.split()[3] + "-KS_SPIN_1-1_0.csr"
                cp2k_settings["Sfile"] = line.split()[3] + "-S_SPIN_1-1_0.csr"
                cp2k_settings["project_name"] = line.split()[3]
            if "KS CSR write|" in line:
                cp2k_settings["n_KS"] = int(line.split()[3])
                cp2k_settings["k_points"] = True
            if "S CSR write|" in line:
                cp2k_settings["n_S"] = int(line.split()[3])
                cp2k_settings["k_points"] = True
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

    if all(key in lattice for key in ("a", "b", "c")):
        lattice["L"] = np.array([lattice["a"], lattice["b"], lattice["c"]])

    at, coords = _read_coordinates(filename)
    if len(at) > 0:
        lattice["at"] = at
        lattice["coords"] = coords

    if "n_KS" in cp2k_settings:
        cp2k_settings["KS_list"] = _read_cell_list(filename, "KS CSR write|", cp2k_settings["n_KS"])

    if "n_S" in cp2k_settings:
        cp2k_settings["S_list"] = _read_cell_list(filename, "S CSR write|", cp2k_settings["n_S"])

    return cp2k_settings, lattice


def _read_coordinates(filename: str, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the atomic coordinates printed by QUICKSTEP in the CP2K output file.

    The block looks like::

           Atom Kind Element         X          Y          Z     Z(eff)    Mass
              1    1 C     6      0.000000   0.000000   0.000000  4.0000  12.0107

    If the file contains several such blocks (e.g. a geometry optimization),
    the last one is kept.

    Parameters
    ----------
    filename : str
        CP2K output file to be read.
    decimals : int
        Number of decimal places to round the coordinates.
    Returns
    -------
    atoms : np.ndarray
        Atom types of the system, empty if the block is not present.
    coords : np.ndarray
        Coordinates of the atoms in angstrom, empty if the block is not present.
    """
    atoms = []
    coords = []

    in_block = False
    with open(filename, "r") as filehandle:
        for line in filehandle:
            if "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM" in line:
                # Restart from the latest block found
                in_block = True
                atoms = []
                coords = []
                continue

            if not in_block:
                continue

            fields = line.split()
            # Blank lines surround the column header, which precedes the atoms
            if len(fields) == 0 or fields[0] == "Atom":
                continue
            # The block ends as soon as a line does not start with an atom index
            if not fields[0].isdigit():
                in_block = False
                continue

            atoms.append(fields[2])
            coords.append(fields[4:7])

    atoms = np.asarray(atoms, dtype=str)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    coords = np.round(coords, decimals=decimals)

    return atoms, coords


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


def read_csr(fname, n=None, complex=False):
    """
    Reads a CP2K CSR file and returns a scipy sparse matrix.

    Parameters
    ----------
    fname : str
        File to be read as '*.csr'
    n : int, optional
        Number of rows/columns in the matrix, by default None
    complex : bool, optional
        Whether the matrix contains complex values, by default False

    Returns
    -------
    M : A scipy sparse matrix containing the matrix in the CSR format.

    """

    val = np.dtype("c16") if complex else np.dtype("f8")
    rec = np.dtype([("h", "i4"), ("row", "i4"), ("col", "i4"), ("val", val), ("t", "i4")])

    M = np.fromfile(fname, dtype=rec)

    M["row"] -= 1
    M["col"] -= 1

    if M["row"].size == 0 or M["col"].size == 0:
        return sparse.csr_matrix((0, 0))

    if n is not None:
        xmax = n
        ymax = n
    else:
        xmax, ymax = np.max(M["row"]), np.max(M["col"])
        xmax += 1
        ymax += 1

    return sparse.csr_matrix((M["val"], (M["row"], M["col"])), shape=(int(xmax), int(ymax)))