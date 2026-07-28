"""Visualization of the lattice and of on-site quantities."""

import numpy as np
from ase import Atoms
from matplotlib import pyplot as plt

from .base import DeviceBase


class VisualizationMixin(DeviceBase):
    """Interactive views and 2D maps of the device."""

    def print_isosurface_self_interaction(self, atom_type: str):
        """
        Print the isosurface self-interaction for a given atom type.

        Parameters:
        -----------
        atom_type : str
            The atom type for which to print the isosurface self-interaction.
        """

        if self.H == {}:
            raise Exception("The Hamiltonian is empty, load a device first")

        at_indices = np.where(self.lattice["at"] == atom_type)[0]
        V_self_inter = np.zeros(len(at_indices))

        H_diag = self.H[(0, 0, 0)].diagonal()

        for n, i in enumerate(at_indices):
            orb_start = self.start_orb_per_at[i]
            orb_end = self.start_orb_per_at[i + 1]
            V_self_inter[n] = np.mean(H_diag[orb_start:orb_end])

        # 2D map of the self-interaction potential over the atomic positions
        x = self.lattice["coords"][at_indices, 0]
        y = self.lattice["coords"][at_indices, 1]

        plt.figure()
        tpc = plt.tripcolor(x, y, V_self_inter, cmap="viridis", shading="gouraud")
        plt.colorbar(tpc, label="V (eV)")
        plt.xlabel("x (Å)")
        plt.ylabel("y (Å)")
        plt.title(f"Self-interaction potential for {atom_type}")
        plt.gca().set_aspect("equal")
        plt.show()
        np.save("self_interaction_" + atom_type + ".npy", V_self_inter)
        np.save("x_" + atom_type + ".npy", x)
        np.save("y_" + atom_type + ".npy", y)

    def visualize_lattice(self):
        """
        Visualize the lattice using the WeasWidget viewer.

        Returns:
        --------
        WeasWidget
            The viewer, to be displayed in a notebook cell.
        """

        symbols = list(self.lattice["at"])
        positions = self.lattice["coords"]
        cell = self.lattice["L"]

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[1, 1, 1])

        from weas_widget import WeasWidget

        viewer = WeasWidget()
        viewer.from_ase(atoms)

        return viewer
