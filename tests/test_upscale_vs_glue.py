"""Gluing two upscaled devices gives the same device as one longer upscale.

Repeating a cell `2 * n` times along a direction, and repeating it `n` times and
gluing a second copy of that to one of its interfaces, describe the same piece
of material. The two devices must therefore hold the same lattice and the same
blocks, up to the order the atoms are stored in: `upscale` lays the cells out
along the direction it repeats, while `glue_other_device` always puts the atoms
of the device being glued first, so both are sorted before being compared.

The only expected difference is periodicity: the upscaled device is still
periodic along the direction, whereas the glued one is not, so its blocks
coupling to the neighbouring cells are gone. Everything in the cell itself has
to match, which is what is checked here.

The test runs on a small tight-binding cell built in the test itself: two atoms
with a different number of orbitals, coupled to their neighbours along x and y,
which is enough to exercise both the blocks along the direction being glued and
the ones perpendicular to it.
"""

import numpy as np
import pytest
from scipy import sparse as sp

from deviceBuilder import Device

# The cell is coupled to its first neighbours along x and y, and isolated along z.
CELL_KEYS = [(i, j, 0) for i in (-1, 0, 1) for j in (-1, 0, 1)]

# Two atoms carrying a different number of orbitals, at positions that are
# distinct in every direction, so that sorting them is unambiguous.
ORB_MAP = {"A": 2, "B": 1}
CELL_AT = np.array(["A", "B"])
CELL_COORDS = np.array([[0.5, 0.4, 0.0], [1.6, 1.1, 0.0]])
CELL_L = np.diag([2.5, 3.0, 10.0]).astype(float)

N_ORB = sum(ORB_MAP[at] for at in CELL_AT)


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


def make_cell() -> Device:
    """
    Build the unit cell every device of this module starts from.

    Returns:
    --------
    Device
        A cell with its Hamiltonian, its overlap and its lattice, coupled to its
        first neighbours along x and y.
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
    dev.coup_available = [1, 1, 0]
    dev.bs_available = True

    dev._refresh_orb_per_at()

    return dev


def in_cell_blocks(blocks: dict, dir: int) -> dict:
    """
    Keep the blocks that do not couple to a neighbouring cell along `dir`.

    Parameters:
    -----------
    blocks : dict
        The Hamiltonian or overlap blocks of a device.
    dir : int
        Direction the device was glued along.
    Returns:
    --------
    dict
        The blocks whose key is zero along `dir`.
    """

    return {key: block for key, block in blocks.items() if key[dir] == 0}


def assert_same_lattice(glued: Device, upscaled: Device):
    """
    Compare the lattice of the glued device with the one of the upscaled device.

    Parameters:
    -----------
    glued, upscaled : Device
        The two devices to compare, with their atoms already sorted.
    """

    np.testing.assert_array_equal(
        glued.lattice["at"], upscaled.lattice["at"], err_msg="the atom types differ"
    )
    np.testing.assert_allclose(
        glued.lattice["coords"],
        upscaled.lattice["coords"],
        atol=1e-12,
        err_msg="the atom positions differ",
    )
    np.testing.assert_allclose(
        glued.lattice["L"],
        upscaled.lattice["L"],
        atol=1e-12,
        err_msg="the lattice vectors differ",
    )
    np.testing.assert_array_equal(
        glued.start_orb_per_at,
        upscaled.start_orb_per_at,
        err_msg="the orbitals are not spread over the atoms in the same way",
    )


def assert_same_blocks(glued: dict, upscaled: dict, dir: int, name: str):
    """
    Compare the in-cell blocks of the glued device with the upscaled ones.

    Parameters:
    -----------
    glued, upscaled : dict
        The Hamiltonian or overlap blocks of the two devices.
    dir : int
        Direction the device was glued along.
    name : str
        Name of the matrix, used in the failure message.
    """

    # Gluing breaks the periodicity along the direction it works on, so the
    # glued device must not be left with any coupling to a neighbouring cell.
    coupled = sorted(key for key in glued if key[dir] != 0)
    assert not coupled, f"{name} still couples to a neighbouring cell along {dir}: {coupled}"

    glued = in_cell_blocks(glued, dir)
    upscaled = in_cell_blocks(upscaled, dir)

    assert glued.keys() == upscaled.keys(), f"{name} holds different blocks"
    for key in sorted(upscaled):
        np.testing.assert_allclose(
            glued[key].toarray(),
            upscaled[key].toarray(),
            atol=1e-12,
            err_msg=f"{name}[{key}] differs",
        )


@pytest.mark.parametrize("interface", [-1, 1], ids=lambda i: f"interface{i:+d}")
@pytest.mark.parametrize("dir", [0, 1], ids="xy".__getitem__)
def test_two_glued_halves_match_the_upscaled_device(dir: int, interface: int):
    n = 2

    # Half the device, with a copy of itself glued to one of its interfaces.
    glued = make_cell()
    glued.upscale(n, dir=dir)
    other = make_cell()
    other.upscale(n, dir=dir)
    glued.glue_other_device(other, dir, interface)

    # The same device, obtained by repeating the cell in one go.
    upscaled = make_cell()
    upscaled.upscale(2 * n, dir=dir)

    # `upscale` and `glue_other_device` do not lay the atoms out in the same
    # order, so both devices are sorted before being compared.
    glued.sort_atoms([0, 1, 2])
    upscaled.sort_atoms([0, 1, 2])

    assert_same_lattice(glued, upscaled)
    assert_same_blocks(glued.H, upscaled.H, dir, "H")
    assert_same_blocks(glued.S, upscaled.S, dir, "S")
