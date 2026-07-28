"""Shared state of a device.

`DeviceBase` holds the attributes every part of a `Device` works on. All the
mixins in this package inherit from it, so they can be type-checked in
isolation and they all share the same view of the device.
"""

import numpy as np


class DeviceBase:
    """The data a device is made of, plus the helpers that keep it consistent."""

    def __init__(self):
        """
        Initialize the Device object.
        """

        self.H = {}  # Dictionary with the Hamiltonian (hopping) matrices
        self.S = {}  # Dictionary with the Overlap matrices
        self.lattice = None  # Dictionary with the lattice information (L, at, coords)

        self.orb_map = None  # Dictionary with number of orbitals per atom type (expressed as string)
        self.start_orb_per_at = (
            None  # List with the starting orbital index for each atom
        )

        self.Fermi = None  # Fermi level of the system

        self.potential = None  # Potential profile over the device

        self.coup_available = [
            0,
            0,
            0,
        ]  # List to check if the coupling in x,y,z is available (not folded into 000 Hamiltonian or not available at all)
        self.bs_available = False  # Boolean to check if the band structure can be computed (i.e. if the couplings in all directions are available)

    def _refresh_orb_per_at(self):
        """
        Recompute the starting orbital index of each atom from the current
        atom list. Must be called after anything that adds, removes or
        reorders atoms.
        """
        self.start_orb_per_at = np.cumsum(
            [0] + [self.orb_map[at] for at in self.lattice["at"]]
        )
