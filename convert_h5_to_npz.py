from os import path

import numpy as np
from mdtraj import io

xyz2 = io.loadh(path.join("test", "data", "frame0.dcd.h5"), "xyz")

np.savez(path.join("test", "data", "frame0.dcd.npz"), dcd_coords=xyz2)
