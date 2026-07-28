"""Writer for the binary matrix format read by OMEN."""

import os

import numpy as np
from scipy import sparse


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
