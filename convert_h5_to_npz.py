from os import path

import numpy as np
from mdtraj import io

xyz2 = io.loadh(path.join("test", "data", "frame0.dcd.h5"), "xyz")
np.savez(path.join("test", "data", "frame0.dcd.npz"), dcd_coords=xyz2)

iofile = io.loadh(path.join("test", "data", "frame0.xtc.h5"), deferred=False)

np.savez(
    path.join("test", "data", "frame0.xtc.npz"), 
    xyz=iofile["xyz"],
    step=iofile["step"],
    box=iofile["box"],
    time=iofile["time"],
)