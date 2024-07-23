##############################################################################
### Original MDTraj disclaimer below:
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with biotraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

"""
The biotraj package contains tools for loading and saving MD trajectories
in different formats, forked from MDTraj.
The mdtraj package contains tools for loading and saving molecular dynamics
trajectories in a variety of formats, including Gromacs XTC & TRR, CHARMM/NAMD
DCD, PDB, and HDF5.
"""

# silence cython related numpy warnings, see github.com/numpy/numpy/pull/432
import numpy as _  # noqa
from .core import element
from .core.topology import Amide, Aromatic, Double, Single, Topology, Triple
from .core.trajectory import (
    Trajectory,
    iterload,
    join,
    load,
    load_frame,
    load_topology,
    open,
)
from .formats.dcd import load_dcd
from .formats.netcdf import load_netcdf
from .formats.pdb import load_pdb
from .formats.prmtop import load_prmtop
from .formats.psf import load_psf
from .formats.registry import FormatRegistry
from .formats.trr import load_trr
from .formats.xtc import load_xtc
from .formats.xyzfile import load_xyz

from .core.trajectory import *

__name__ = "biotraj"
__all__ = (
    "element",
    "Amide",
    "Aromatic",
    "Double",
    "Single",
    "Topology",
    "Triple",
    "load_dcd",
    "load_lammpstrj",
    "load_netcdf",
    "load_pdb",
    "load_prmtop",
    "load_psf",
    "load_trr",
    "load_xtc",
    "load_xyz",
    "FormatRegistry",
    "open",
    "load",
    "iterload",
    "load_frame",
    "load_topology",
    "join",
    "Trajectory",
)


def capi():
    import os
    import sys

    module_path = sys.modules["biotraj"].__path__[0]
    return {
        "lib_dir": os.path.join(module_path, "core", "lib"),
        "include_dir": os.path.join(module_path, "core", "lib"),
    }
