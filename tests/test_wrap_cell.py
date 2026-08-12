"""Wrapping a device that was translated onto itself gives the device back.

Translating a device by a vector that maps its lattice onto itself, wrapping the
atoms that left the cell back into it and sorting them again describes the very
same piece of material: the atoms are the ones one started from, only relabelled,
so the lattice and the blocks have to come back unchanged.

Wrapping an atom by a lattice vector changes the cell the elements it appears in
belong to: `<i,0|H|j,R>` is stored at `R + t_i - t_j` once the atoms `i` and `j`
have been moved by `t_i` and `t_j`. When atoms are wrapped by different vectors,
an element can leave the range of the stored couplings while only part of the
translations have been applied, and come back into it once the others are, so it
must not be dropped on the way. This is why the shift is exercised along x and y
at the same time, and not only along one direction at a time: with a single
translation nothing ever leaves the range.

The test runs on a small tight-binding cell built in the test itself: two atoms
with a different number of orbitals, repeated four times along x and y. The cell
reaches its third neighbours along x, so that once it is repeated the atoms on
one side of the device still couple to the ones on the other side of the
neighbouring one: those are the elements that travel the furthest when the
device is wrapped, and the ones that used to be dropped.
"""

import numpy as np
import pytest
from scipy import sparse as sp

from deviceBuilder import Device

# The cell reaches its third neighbours along x and its first ones along y, and
# is isolated along z.
CELL_KEYS = [(i, j, 0) for i in range(-3, 4) for j in (-1, 0, 1)]

# Two atoms carrying a different number of orbitals, at positions that are
# distinct in every direction, so that sorting them is unambiguous.
ORB_MAP = {"A": 2, "B": 1}
CELL_AT = np.array(["A", "B"])
CELL_COORDS = np.array([[0.5, 0.4, 0.0], [1.6, 1.1, 0.0]])
CELL_L = np.diag([2.5, 3.0, 10.0]).astype(float)

N_ORB = sum(ORB_MAP[at] for at in CELL_AT)

# Number of repetitions of the cell along x and y. The device is shifted by a
# whole number of cells, so that the lattice ends up on the one it started from.
N_REPEAT = 4


def hermitian_blocks(seed: int, shift: float) -> dict:
    """
    Build one set of blocks, filled with random numbers.

    The block of every coupling is the transpose of the one of the opposite
    coupling, and the in-cell block is symmetric, as in a device read from a
    real calculation.

    Parameters:
    -----------
    seed : int
        Seed of the generator, so that a run is reproducible.
    shift : float
        Value added to the diagonal of the in-cell block.
    Returns:
    --------
    dict
        One `csr_matrix` per key of `CELL_KEYS`.
    """

    rng = np.random.default_rng(seed)
    blocks = {}

    # Draw one half of the couplings, and get the other half by transposition.
    for key in (key for key in CELL_KEYS if key > (0, 0, 0)):
        block = rng.normal(scale=0.5, size=(N_ORB, N_ORB))
        blocks[key] = sp.csr_matrix(block)
        blocks[tuple(-index for index in key)] = sp.csr_matrix(block.T)

    in_cell = rng.normal(scale=0.5, size=(N_ORB, N_ORB))
    blocks[(0, 0, 0)] = sp.csr_matrix(in_cell + in_cell.T + shift * np.eye(N_ORB))

    return blocks


def make_device() -> Device:
    """
    Build the device every case of this module starts from.

    Returns:
    --------
    Device
        The unit cell repeated `N_REPEAT` times along x and y, with its atoms
        already sorted.
    """

    dev = Device()

    dev.H = hermitian_blocks(seed=0, shift=4.0)
    dev.S = hermitian_blocks(seed=1, shift=8.0)

    dev.lattice = {
        "L": CELL_L.copy(),
        "at": CELL_AT.copy(),
        "coords": CELL_COORDS.copy(),
    }
    dev.orb_map = dict(ORB_MAP)
    dev.Fermi = 0.0

    # The couplings along x and y are kept apart from the in-cell block, the
    # ones along z are not there at all.
    dev._update_max_coupling()


    dev._refresh_orb_per_at()

    dev.upscale(N_REPEAT, dir=0)
    dev.upscale(N_REPEAT, dir=1)
    dev.sort_atoms([0, 1, 2])

    dev.PBC = [True, True, False]  # The device is periodic along x and y, but not along z

    return dev


def assert_same_lattice(wrapped: Device, reference: Device):
    """
    Compare the lattice of the wrapped device with the one it started from.

    Parameters:
    -----------
    wrapped, reference : Device
        The two devices to compare, with their atoms already sorted.
    """

    np.testing.assert_array_equal(
        wrapped.lattice["at"], reference.lattice["at"], err_msg="the atom types differ"
    )
    np.testing.assert_allclose(
        wrapped.lattice["coords"],
        reference.lattice["coords"],
        atol=1e-12,
        err_msg="the atom positions differ",
    )
    np.testing.assert_allclose(
        wrapped.lattice["L"],
        reference.lattice["L"],
        atol=1e-12,
        err_msg="the lattice vectors differ",
    )
    np.testing.assert_array_equal(
        wrapped.start_orb_per_at,
        reference.start_orb_per_at,
        err_msg="the orbitals are not spread over the atoms in the same way",
    )


def assert_same_blocks(wrapped: dict, reference: dict, name: str):
    """
    Compare the blocks of the wrapped device with the ones it started from.

    Both devices are padded with empty blocks up to the range of their
    couplings, so only the blocks that are there on either side are compared,
    a missing one counting as empty.

    Parameters:
    -----------
    wrapped, reference : dict
        The Hamiltonian or overlap blocks of the two devices.
    name : str
        Name of the matrix, used in the failure message.
    """

    # An element leaving the range of the couplings while the device is being
    # wrapped is silently dropped, so the elements are counted before being
    # compared: it tells a loss apart from a block that simply moved.
    stored = sum(block.nnz for block in wrapped.values())
    expected = sum(block.nnz for block in reference.values())
    assert stored == expected, f"{name} lost {expected - stored} elements while wrapping"

    empty = sp.csr_matrix(reference[(0, 0, 0)].shape)
    for key in sorted(set(wrapped) | set(reference)):
        np.testing.assert_allclose(
            wrapped.get(key, empty).toarray(),
            reference.get(key, empty).toarray(),
            atol=1e-12,
            err_msg=f"{name}[{key}] differs",
        )


# The shifts are given in units of the device, and are a whole number of cells:
# half of it along x, a quarter of it along y. Shifting along both directions at
# the same time is the case that matters, since the atoms are then wrapped by
# three different vectors, and the elements coupling two of those atoms have to
# be carried by two translations before they land on their block.
@pytest.mark.parametrize(
    "shift",
    [(0.5, 0.0, 0.0), (0.0, 0.25, 0.0), (0.5, 0.25, 0.0)],
    ids=["along-x", "along-y", "along-xy"],
)
def test_wrapping_a_translated_device_leaves_it_unchanged(shift: tuple):
    reference = make_device()

    # The shift is a whole number of cells, so the shifted lattice sits on the
    # one it started from once the atoms that left the cell are wrapped back.
    wrapped = make_device()
    wrapped.translate_cell(np.array(shift) @ wrapped.lattice["L"])
    wrapped.wrap_cell()
    wrapped.sort_atoms([0, 1, 2])

    assert_same_lattice(wrapped, reference)
    assert_same_blocks(wrapped.H, reference.H, "H")
    assert_same_blocks(wrapped.S, reference.S, "S")
