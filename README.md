# deviceBuilder

Build transport-ready devices out of DFT output.

`deviceBuilder` loads tight-binding-like Hamiltonian and overlap blocks from CP2K,
Wannier90 or plain arrays, and gives you the operations needed to turn a small
periodic unit cell into a full device: extract the couplings between cells, sort
and trim the lattice, repeat the cell, glue different regions together, add a
potential profile, and write out the input decks for QUATREX or OMEN.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, plus `numpy`, `scipy`, `ase` and `matplotlib`. A few
features pull in extra packages only when you call them — see
[Optional dependencies](#optional-dependencies).

## Quick example

Take a CP2K calculation of a small unit cell, stretch it into a four-cell
channel with a potential barrier in the middle, and export it:

```python
from deviceBuilder import Device

dev = Device()

# Read the lattice, the KS and the overlap matrices out of the CP2K run.
# At Gamma the couplings between neighbouring cells are cut out of the Gamma
# matrix; with K_POINTS they are read straight from the .csr files.
dev.load_from_cp2k("cp2k.out", path="./scf")

# Order atoms and orbitals along x, then y, then z
dev.sort_atoms([0, 1, 2])

# Repeat the cell four times along the transport direction
dev.upscale(4, dir=0)

# A 0.4 eV barrier in the middle third of the device
dev.generate_potential_barrier(0.0, 0.4, 0.0, slope=2.0, dir=0)

# Write device/inputs/{hamiltonian.h5,overlap.h5,structure.xyz,potential.npy}
# and device/config.toml, with a left and a right contact
dev.export_data_QUATREX(transport_dir=[0], output_dir="device", cont="lr")
```

Check the band structure of the cell before building the device:

```python
dev = Device()
dev.load_from_cp2k("cp2k.out", path="./scf")

# Gamma -> X, 40 points, plotted in a window of ±1 eV around the Fermi level
eig, cond_min, val_max = dev.compute_band_structure(
    [[0, 0, 0], [1, 0, 0]], n_point=40, dE=1.0
)
print("gap:", cond_min - val_max)
```

## How a device is represented

A `Device` is a set of sparse blocks indexed by cell, plus the lattice they live on:

| attribute | meaning |
| --- | --- |
| `H`, `S` | `dict` of `scipy.sparse.csr_matrix`, keyed by cell index |
| `lattice` | `dict` with `"L"` (3×3 lattice vectors, as rows), `"at"` (atom types), `"coords"` (N×3, Cartesian) |
| `orb_map` | number of orbitals per atom type, e.g. `{"C": 13, "H": 5}` |
| `start_orb_per_at` | first orbital index of each atom, derived from the two above |
| `Fermi` | Fermi level, in eV |
| `potential` | potential profile over the atoms, or `None` |
| `coup_available` | per direction: is the inter-cell coupling still present? |
| `bs_available` | whether a band structure can still be computed |

`H[(i, j, k)]` is the block coupling the home cell to the cell at
`i·a₁ + j·a₂ + k·a₃`; `H[(0, 0, 0)]` is the cell itself. **All energies are in
eV** — matrices read from CP2K are converted from Hartree on load.

The loaders leave the keys of `H` and `S` filling a full box: if the furthest
coupling reaches `n` cells along a direction, every index from `-n` to `n` is
present, the ones with nothing in them holding an all-zero matrix.

Operations that add, remove or reorder atoms keep `start_orb_per_at` in sync for
you. Operations that discard couplings (`upscale_cont`, `glue_other_device`)
clear the corresponding `coup_available` flag, so a later
`compute_band_structure` fails loudly rather than returning a wrong answer.

## API

### Loading

| method | source |
| --- | --- |
| `load_from_cp2k(filename, path, eps)` | a CP2K run, at Gamma or with `K_POINTS` |
| `load_from_wannier90(filename, path, eps, gather_wannier_centers)` | `seedname.wout` + `seedname_hr.dat` |
| `load_from_MALOQ_GAMMA(H_file, S_file, xyz_file, orb_map, eps)` | H and S as `.npy`/`.npz`, geometry from an xyz file |

`filename` is the CP2K output file and `path` the directory holding it. Lattice
vectors, atomic coordinates and the number of orbitals per kind are parsed out
of the output itself, and the matrix files it refers to are read next to it —
nothing else has to be pointed at. Whether the run used `K_POINTS` is detected
from the same file:

* **Gamma** — the KS and overlap matrices are read, then the couplings are cut
  out of them along each of the three directions in turn. A direction is only
  kept if the in-cell and coupling blocks come out cleanly separated; when they
  do not, a message says so and that direction is left folded into
  `H[(0, 0, 0)]`, with `coup_available` still `0`. That usually means the cell
  is too small along it.
* **`K_POINTS`** — one `.csr` per cell index is read as is, and all three
  directions are available.

`eps` drops matrix elements below a threshold (the overlap uses `eps/10`). Only
`load_from_MALOQ_GAMMA` needs an explicit `orb_map`.

Neither the Wannier90 nor the MALOQ loader knows the Fermi level: the first sets
it to `-999` and warns, the second to `0`. Assign `dev.Fermi` yourself before
exporting, since the export writes energy windows around it.

### Checks done on load

Every loader ends by putting the blocks in a canonical, verified state, so a
malformed run fails at load time rather than somewhere downstream:

* if the blocks turn out to be stored upper-triangular only — as CP2K does for a
  symmetric matrix — the missing halves are filled in from the transpose, and a
  message reports it;
* `H` and `S` are checked to be Hermitian: every block must have its opposite
  `(-i, -j, -k)`, of the same shape and equal to its transpose;
* the set of cell indices is padded with empty blocks up to the full box
  described above.

### Reshaping the lattice

| method | effect |
| --- | --- |
| `sort_atoms(dir=[0,1,2], decimals=6)` | lexsort atoms and orbitals, primary key `dir[0]` |
| `upscale(n, dir=0)` | repeat the cell `n` times along `dir`, rebuilding all blocks |
| `remove_atoms(b1, b2)` | drop every atom in the box `[b1, b2)`, with its orbitals |
| `center_cell()` | center the atoms in the unit cell |
| `align_with_axis()` | move the lattice corner to the origin |
| `translate_cell(shift)` | rigid translation by a Cartesian vector |
| `shift_energies(dE)` | rigid shift of H and of the Fermi level, in eV |
| `reduce_to_gamma(dir=0)` | fold all transverse couplings into the blocks along `dir` |

### Assembling a device

```python
channel.glue_other_device(contact, dir=0, interface=1, tol=0.1)
```

attaches `contact` to the positive end of `channel` along x (`interface=-1` for
the negative end). Atoms within `tol` of each other at the seam are matched up,
their types are checked, and the periodic coupling blocks of `contact` become
the coupling between the two regions.

```python
dev.upscale_cont(orig, vec, dir=0, n=3, compute_band=False)
```

repeats a contact cell — the one with origin `orig` and lattice vectors `vec`
(3×3) — `n` times on the negative side of `dir`. The cell must sit at the
negative edge of the device and must not couple beyond its immediate neighbour;
both are checked. With `compute_band=True` the contact band structure is plotted
at transverse Gamma, which is a good sanity check on `orig` and `vec`.

### Potential, bands and views

| method | |
| --- | --- |
| `generate_potential_barrier(V1, V2, V3, slope, dir, grid_from_OMEN=None)` | three regions joined by error functions, stored in `dev.potential` |
| `compute_band_structure(k_path, n_point=10, dE=2.0)` | returns `(eigenvalues, conduction_min, valence_max)` and plots the bands |
| `visualize_lattice()` | interactive viewer, to be returned from a notebook cell |
| `print_isosurface_self_interaction(atom_type)` | 2D map of the mean on-site energy per atom of one type |

### Exporting

| method | writes |
| --- | --- |
| `export_data_QUATREX(transport_dir=[0], output_dir="device", cont="lr")` | `inputs/hamiltonian.h5`, `inputs/overlap.h5`, `inputs/structure.xyz`, `inputs/potential.npy` and `config.toml` |
| `export_data_OMEN(output_dir="device_OMEN")` | `H_*.bin`, `S_*.bin`, `lattice_dat`, `vact_dat` |

Blocks that couple along a transport direction are left out of the QUATREX
export — the transport code builds those itself from the contacts. `cont` selects
which contacts to describe in `config.toml`: any combination of `"l"`, `"r"`,
`"t"`, `"b"`.

## Optional dependencies

| package | needed for |
| --- | --- |
| `toml`, `qttools` | `export_data_QUATREX` |
| `weas_widget` | `visualize_lattice` |
| `cupy` | GPU-accelerated `compute_band_structure`; falls back to NumPy when absent |

## Tests

```bash
pip install -e ".[test]"
pytest
```

Each directory under [examples/](examples/) holds a build script next to the CP2K
run it reads and the deck it exported; the suite reruns every one of them in a
temporary directory and compares the deck file by file, so dropping a new
example in is enough to have it covered.

## License

MIT — see [LICENSE](LICENSE).
