from os.path import join
import functools
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

import biotraj as md
import biotraj.core.trajectory
import biotraj.formats
import biotraj.utils
from biotraj.core import element

from .util import data_dir

TrajObj = namedtuple("TrajObj", ["fobj", "fext", "fn"])
file_objs = [
    (md.formats.NetCDFTrajectoryFile, "nc"),
    (md.formats.XTCTrajectoryFile, "xtc"),
    (md.formats.TRRTrajectoryFile, "trr"),
    (md.formats.DCDTrajectoryFile, "dcd"),
    (md.formats.PDBTrajectoryFile, "pdb"),
    (md.formats.PDBTrajectoryFile, "pdb.gz"),
    (md.formats.GroTrajectoryFile, "gro"),
]

# Handle reference trajectories
@pytest.fixture(params=file_objs, ids=lambda x: x[1])
def ref_traj(request, monkeypatch):
    fobj, fext = request.param

    return TrajObj(fobj, fext, f"frame0.{fext}")

# Create trajectory object for different file formats
@pytest.fixture(params=file_objs, ids=lambda x : x[1])
def write_traj(request, tmpdir):
    fobj, fext = request.param
    return TrajObj(fobj, fext, f"{tmpdir}/traj.{fext}")

# Omit formats without box information 
# (Currently only PDB, extend if necessary)
@pytest.fixture
def write_traj_with_box(write_traj):
    if write_traj.fext in ["pdb", "pdb.gz"]:
        pytest.skip(f"{write_traj.fext} does not store box information")
    else:
        return write_traj

@pytest.fixture(scope="module")
def xtc_path():
    return join(data_dir(), "frame0.xtc")    

def data_path(file_str):
    return join(data_dir(), file_str)

# Some formats don't save time information
def has_time_info(fext):
    return fext not in [
        "dcd",
        "pdb",
        "pdb.gz"
    ]

# "xyz", "lammpstrj", "lh5" -> 3, 6 otherwise
# NOTE: Keep in case xyz/lammpstrj are readded again
def precision(fext):
    #if fext in ["xyz", "lammpstrj", "lh5"]:
    #    return 3
    #else:
    #    return 6
    return 6

def precision2(fext1, fext2):
    return min(precision(fext1), precision(fext2))

# Load trajectory with larger topology -> Test raised error
# (22 atoms in XTC v. 2_000 atoms)
def test_mismatch(xtc_path, larger_topol_path=data_path("4ZUO.pdb")):
    with pytest.raises(ValueError):
        md.load(xtc_path, top=larger_topol_path)

# Test correct handling of boxes after manual addition to PDB w/o box
def test_box(pdb_path = data_path("native.pdb")):
    t = md.load(pdb_path)
    assert t.unitcell_vectors is None
    assert t.unitcell_lengths is None
    assert t.unitcell_angles is None
    assert t.unitcell_volumes is None
    t.unitcell_vectors = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ).reshape(1, 3, 3)
    assert np.allclose(np.array([1.0, 1.0, 1.0]), t.unitcell_lengths[0])
    assert np.allclose(np.array([90.0, 90.0, 90.0]), t.unitcell_angles[0])
    assert np.allclose(np.array([1.0]), t.unitcell_volumes)

# Test, whether small boxes are read as is for PDB files
# (unreasonably small boxes are removed heuristically with standard settings).
def test_load_pdb_box(pdb_path = data_path("native2.pdb")):
    t = md.load(pdb_path, no_boxchk=True)
    assert np.allclose(t.unitcell_lengths[0], np.array([0.1, 0.2, 0.3]))
    assert np.allclose(t.unitcell_angles[0], np.array([90.0, 90.0, 90.0]))
    assert np.allclose(
        t.unitcell_vectors[0], 
        np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    )

# Check whether box information is preserved over load/save
# (PDB -> Traj formats)
def test_box_load_save(
        write_traj_with_box,
        pdb_path = data_path("native.pdb"), 
        pdb_path_2 = data_path("native2.pdb")
        ):
    t = md.load(pdb_path_2, no_boxchk=True)
    top = md.load_topology(pdb_path, no_boxchk=True)

    t.save(write_traj_with_box.fn)
    t2 = md.load(write_traj_with_box.fn, top=top)

    assert t.unitcell_vectors is not None
    assert np.allclose(t.xyz, t2.xyz)
    assert np.allclose(t.unitcell_vectors, t2.unitcell_vectors)
    assert np.allclose(t.unitcell_angles, t2.unitcell_angles)
    assert np.allclose(t.unitcell_lengths, t2.unitcell_lengths)

# Test slice (load prepared XTC file)
def test_slice(
        intraj_prep=data_path("traj_prep.xtc"),
        intraj_prep_top=data_path("traj_prep_top.pdb")
):
    t = md.load(intraj_prep, top=intraj_prep_top)

    # with copying
    assert np.allclose((t[0:5] + t[5:10]).xyz, t[0:10].xyz)
    assert np.allclose((t[0:5] + t[5:10]).time, t[0:10].time)
    assert np.allclose((t[0:5] + t[5:10]).unitcell_vectors, t[0:10].unitcell_vectors)
    assert np.allclose((t[0:5] + t[5:10]).unitcell_lengths, t[0:10].unitcell_lengths)
    assert np.allclose((t[0:5] + t[5:10]).unitcell_angles, t[0:10].unitcell_angles)

    # without copying (in place)
    assert np.allclose(
        (t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)).xyz,
        t.slice(key=range(10), copy=False).xyz,
    )
    assert np.allclose(
        (t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)).time,
        t.slice(key=range(10), copy=False).time,
    )
    assert np.allclose(
        (t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)).unitcell_vectors,
        t.slice(key=range(10), copy=False).unitcell_vectors,
    )
    assert np.allclose(
        (t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)).unitcell_lengths,
        t.slice(key=range(10), copy=False).unitcell_lengths,
    )
    assert np.allclose(
        (t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)).unitcell_angles,
        t.slice(key=range(10), copy=False).unitcell_angles,
    )

def test_slice_2(
        intraj_prep=data_path("traj_prep.xtc"),
        intraj_prep_top=data_path("traj_prep_top.pdb")
):
    t = md.load(intraj_prep, top=intraj_prep_top)
    
    # with copying
    assert t[0] == t[[0, 1]][0]
    # without copying (in place)
    assert t.slice(key=0, copy=False) == t.slice(key=[0, 1], copy=True)[0]

def test_read_path(ref_traj, data_path("native.pdb"), monkeypatch):
    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            md.load(Path(get_fn(ref_traj.fn)), top=get_fn("native.pdb"))

    md.load(Path(get_fn(ref_traj.fn)), top=get_fn("native.pdb"))   