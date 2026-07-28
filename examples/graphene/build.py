from deviceBuilder import Device

# Build the channel from the graphene unit cell computed by CP2K with K_POINTS.
# "graph.out" is the CP2K output file, "SCF" the directory containing it (and the
# .csr matrices / coordinate file it refers to).
dev1 = Device()
dev1.load_from_cp2k_K_POINTS("graph.out", "SCF")

# Repeat the unit cell 6 times along x, folding the periodic couplings into the
# enlarged cell.
dev1.upscale(6, 0)

# Second, identical block: it will be glued on both sides of dev1 to act as the
# left and right leads.
dev2 = Device()
dev2.load_from_cp2k_K_POINTS("graph.out", "SCF")

dev2.upscale(6, 0)

# Attach dev2 on the negative (-1) and positive (+1) x interfaces of dev1.
# dev1 now spans 3 blocks along the transport direction.
dev1.glue_other_device(dev2, 0, -1)
dev1.glue_other_device(dev2, 0, 1)

# Rigidly translate the atomic coordinates (in Angstrom, Cartesian).
import numpy as np

dev1.translate_cell(np.array([1, 1, 1]))

# Rigid shift of the Hamiltonian (H += S * dE), the Fermi level follows.
dev1.shift_energies(0.5)

# Reorder atoms/orbitals lexicographically in fractional coordinates so that the
# matrices come out block-tridiagonal along the transport direction. `decimals`
# sets the rounding used to consider coordinates equal.
dev1.sort_atoms(decimals=2)

# Write the QUATREX inputs (default: transport along x, left/right contacts) to
# ./device/inputs.
dev1.export_data_QUATREX()