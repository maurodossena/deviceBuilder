from deviceBuilder import Device

dev1 = Device()
dev1.load_from_cp2k("graph.out","SCF")

dev1.upscale(6,0)

dev2 = Device()
dev2.load_from_cp2k("graph.out","SCF")

dev2.upscale(6,0)

print(dev1.H)

dev1.glue_other_device(dev2, 0, -1)
dev1.glue_other_device(dev2, 0, 1)

import numpy as np
dev1.translate_cell(np.array([1,1,1]))

dev1.shift_energies(0.5)

dev1.sort_atoms(decimals=2)

dev1.export_data_QUATREX()
