from deviceBuilder import Device

# Build the channel from the graphene unit cell computed by CP2K with K_POINTS.
# "graph.out" is the CP2K output file, "SCF" the directory containing it (and the
# .csr matrices / coordinate file it refers to).
dev1 = Device()
dev1.load_from_cp2k_GAMMA("diam.out", "SCF",extract_x=True, extract_y=True, extract_z=True)

# Repeat the unit cell 6 times along x, folding the periodic couplings into the
# enlarged cell.
dev1.upscale(2, 0)

import numpy as np

dev1.translate_cell(np.array([0.2, 0.5, 0.3]))

# Rigid shift of the Hamiltonian (H += S * dE), the Fermi level follows.
dev1.shift_energies(0.3)

# Reorder atoms/orbitals lexicographically in fractional coordinates so that the
# matrices come out block-tridiagonal along the transport direction. `decimals`
# sets the rounding used to consider coordinates equal.
dev1.sort_atoms(decimals=2)

# Write the QUATREX inputs (default: transport along x, left/right contacts) to
# ./device/inputs.
dev1.export_data_QUATREX()