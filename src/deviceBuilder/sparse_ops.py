"""Small operations on the sparse blocks that make up a device."""

from scipy import sparse
import numpy as np


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


def prune_small(block: sparse.csr_matrix, eps: float) -> sparse.csr_matrix:
    """
    Drop, in place, all the elements whose absolute value is below a threshold.

    Parameters
    ----------
    block : sparse.csr_matrix
        Matrix to be pruned.
    eps : float
        Threshold below which matrix elements are set to zero.
    
    Returns
    -------
    sparse.csr_matrix
        Pruned matrix.
    """

    block.data[np.absolute(block.data) <= eps] = 0
    block.eliminate_zeros()

    return block
