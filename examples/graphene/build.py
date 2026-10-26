from deviceBuilder import (
    export_data_QUATREX,
    glue_other_device,
    load_from_cp2k,
    shift_energies,
    sort_atoms,
    translate_cell,
    upscale,
)
import numpy as np

dev1 = load_from_cp2k("graph.out","SCF")
translate_cell(dev1, np.array([0.5,0.5,0.5])@dev1.lattice["L"])

upscale(dev1, 6,0)

sort_atoms(dev1, [1,0,2],decimals=2)

dev2 = load_from_cp2k("graph.out","SCF")

translate_cell(dev2, np.array([0.5,0.5,0.5])@dev2.lattice["L"])

upscale(dev2, 6,0)

glue_other_device(dev1, dev2, 0, -1)
glue_other_device(dev1, dev2, 0, 1)

translate_cell(dev1, np.array([1,1,1]))

shift_energies(dev1, 0.5)

sort_atoms(dev1, decimals=2)

export_data_QUATREX(dev1)
