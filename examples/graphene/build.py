from deviceBuilder import Device
import numpy as np

dev1 = Device()
dev1.load_from_cp2k("graph.out","SCF")
dev1.translate_cell(np.array([0.5,0.5,0.5])@dev1.lattice["L"])

dev1.upscale(6,0)

dev1.sort_atoms([1,0,2],decimals=2)

dev2 = Device()
dev2.load_from_cp2k("graph.out","SCF")

dev2.translate_cell(np.array([0.5,0.5,0.5])@dev2.lattice["L"])

dev2.upscale(6,0)

dev1.glue_other_device(dev2, 0, -1)
dev1.glue_other_device(dev2, 0, 1)

dev1.translate_cell(np.array([1,1,1]))

dev1.shift_energies(0.5)

dev1.sort_atoms(decimals=2)

dev1.export_data_QUATREX()
