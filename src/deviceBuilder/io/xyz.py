"""Reader for extended XYZ files."""

import numpy as np


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
