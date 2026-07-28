"""Physical constants and unit conversions used across the package."""

# Conversion applied to Hamiltonian matrix elements read from CP2K.
HARTREE_TO_EV = 27.2114079527

# Lower-precision conversion used for energies printed in CP2K output files
# (Fermi level, eigenvalues). Kept separate from HARTREE_TO_EV so that values
# parsed from output files stay identical to previous versions of the package.
HARTREE_TO_EV_OUTPUT = 27.2114
