from deviceBuilder import (
    export_data_QUATREX,
    load_from_cp2k,
    shift_energies,
    sort_atoms,
    translate_cell,
    upscale,
)
import numpy as np

# Build the channel from the diamond unit cell computed by CP2K at Gamma, so the
# couplings are cut out of the Gamma-point matrices along each direction.
# "diam.out" is the CP2K output file, "SCF" the directory containing it (and the
# .csr matrices / coordinate file it refers to).
dev1 = load_from_cp2k("diam.out", "SCF")

# Repeat the unit cell 2 times along x, folding the periodic couplings into the
# enlarged cell.
upscale(dev1, 2, 0)

translate_cell(dev1, np.array([0.2, 0.5, 0.3]))

# Rigid shift of the Hamiltonian (H += S * dE), the Fermi level follows.
shift_energies(dev1, 0.3)

# Reorder atoms/orbitals lexicographically in fractional coordinates so that the
# matrices come out block-tridiagonal along the transport direction. `decimals`
# sets the rounding used to consider coordinates equal.
sort_atoms(dev1, decimals=2)

# Write the QUATREX inputs (default: transport along x, left/right contacts) to
# ./device/inputs.
export_data_QUATREX(dev1)