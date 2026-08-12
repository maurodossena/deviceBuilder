from deviceBuilder import Device
import numpy as np

# Build the channel from the diamond unit cell computed by CP2K at Gamma, so the
# couplings are cut out of the Gamma-point matrices along each direction.
# "diam.out" is the CP2K output file, "SCF" the directory containing it (and the
# .csr matrices / coordinate file it refers to).
dev1 = Device()
dev1.load_from_cp2k("diam.out", "SCF")

# Repeat the unit cell 2 times along x, folding the periodic couplings into the
# enlarged cell.
dev1.upscale(2, 0)

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