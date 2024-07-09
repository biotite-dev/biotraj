from os.path import join

import numpy as np
import pytest

from biotraj.formats import DCDTrajectoryFile

from .util import data_dir

def test_read():
    dcd_path = join(data_dir(), "frame0.dcd")
    dcd_npz_reference_path = join(data_dir(), "frame0.dcd.npz")
    xyz, box_lengths, box_angles = DCDTrajectoryFile(dcd_path).read()
    xyz2 = np.load(dcd_npz_reference_path)["dcd_coords"]

    assert np.allclose(xyz, xyz2)