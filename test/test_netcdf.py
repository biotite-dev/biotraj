import os
from os.path import join
import subprocess
import tempfile
import sys
from distutils.spawn import find_executable

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

import biotraj as md
from biotraj.formats import NetCDFTrajectoryFile
from .util import data_dir

@pytest.fixture(scope="module")
def nc_path():
    return join(data_dir(), "mdcrd.nc")

@pytest.fixture(scope="module")
def pdb_path():
    return join(data_dir(), "native.pdb")

needs_cpptraj = pytest.mark.skipif(
    find_executable("cpptraj") is None,
    reason=(
        "This test requires cpptraj from AmberTools to be installed " 
        "(http://ambermd.org)."
        "Alternatively, a Conda package is also available."
    ),
)

fd, temp = tempfile.mkstemp(suffix=".nc")
fd2, temp2 = tempfile.mkstemp(suffix=".nc")

class TestNetCDFNetCDF4():
    """
    This class contains all the tests that we would also want to run with scipy.
    Now a class so we can subclass it for later.
    """
    def teardown_module(self, module):
        """remove the temporary file created by tests in this file
        this gets automatically called by pytest"""
        os.close(fd)
        os.close(fd2)
        os.unlink(temp)
        os.unlink(temp2)
    
    def test_read_after_close(self, nc_path):
        """Default test using netCDF4"""
        f = NetCDFTrajectoryFile(nc_path)
        assert np.allclose(f.n_atoms, 223)
        assert np.allclose(f.n_frames, 101)
    
        f.close()
    
        # should be an IOError if you read a file that's closed
        with pytest.raises(IOError):
            f.read()
    

    
    def test_read_chunk_1(self, nc_path):
        """Default test using netCDF4"""
        with NetCDFTrajectoryFile(nc_path) as file:
            a, b, c, d = file.read(10)
            e, f, g, h = file.read()
    
            assert np.allclose(len(a), 10)
            assert np.allclose(len(b), 10)
    
            assert np.allclose(len(e), 101 - 10)
            assert np.allclose(len(f), 101 - 10)
    
        with NetCDFTrajectoryFile(nc_path) as file:
            xyz = file.read()[0]
    
        assert np.allclose(a, xyz[0:10])
        assert np.allclose(e, xyz[10:])
    
    def test_read_chunk_2(self, nc_path):
        """Default test using netCDF4"""
    
        with NetCDFTrajectoryFile(nc_path) as file:
            a, b, c, d = file.read(10)
            e, f, g, h = file.read(100000000000)
    
            assert np.allclose(len(a), 10)
            assert np.allclose(len(b), 10)
    
            assert np.allclose(len(e), 101 - 10)
            assert np.allclose(len(f), 101 - 10)
    
        with NetCDFTrajectoryFile(nc_path) as file:
            xyz = file.read()[0]
    
        assert np.allclose(a, xyz[0:10])
        assert np.allclose(e, xyz[10:])
    
    def test_read_chunk_3(self, nc_path):
        """Default test using netCDF4"""
        # too big of a chunk should not be an issue
        with NetCDFTrajectoryFile(nc_path) as file:
            a = file.read(1000000000)
        with NetCDFTrajectoryFile(nc_path) as file:
            b = file.read()
    
        assert np.allclose(a[0], b[0])
    
    def test_read_write_1(self):
        """Default test using netCDF4"""
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
        boxlengths = np.random.randn(100, 3)
        boxangles = np.random.randn(100, 3)
    
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time, boxlengths, boxangles)
    
        with NetCDFTrajectoryFile(temp) as f:
            a, b, c, d = f.read()
            assert np.allclose(a, xyz)
            assert np.allclose(b, time)
            assert np.allclose(c, boxlengths)
            assert np.allclose(d, boxangles)
    
    def test_read_write_2(self, pdb_path):
        """Default test using netCDF4"""
        xyz = np.random.randn(5, 22, 3)
        time = np.random.randn(5)
    
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time)
    
        with NetCDFTrajectoryFile(temp) as f:
            rcoord, rtime, rlengths, rangles = f.read()
            assert np.allclose(rcoord, xyz)
            assert np.allclose(rtime, time)
            assert rlengths is None
            assert rangles is None
    
        t = md.load(temp, top=pdb_path)
        
        # Convert array to float: None -> NaN
        assert np.all(np.isnan(np.array(t.unitcell_angles, dtype=float)))
        assert np.all(np.isnan(np.array(t.unitcell_lengths, dtype=float)))
    
    def test_ragged_1(self):
        """Default test using netCDF4"""
        # try first writing no cell angles/lengths, and then adding some
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
        cell_lengths = np.random.randn(100, 3)
        cell_angles = np.random.randn(100, 3)
    
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time)
            with pytest.raises(ValueError):
                f.write(xyz, time, cell_lengths, cell_angles)
    
    def test_ragged_2(self):
        """Default test using netCDF4"""
        # try first writing no cell angles/lengths, and then adding some
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
        cell_lengths = np.random.randn(100, 3)
        cell_angles = np.random.randn(100, 3)
    
        # from mdtraj.formats import HDF5TrajectoryFile
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time, cell_lengths, cell_angles)
            with pytest.raises(ValueError):
                f.write(xyz, time)
    
    def test_read_write_25(self):
        """Default test using netCDF4"""
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
    
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time)
            f.write(xyz, time)
    
        with NetCDFTrajectoryFile(temp) as f:
            a, b, c, d = f.read()
            assert np.allclose(a[0:100], xyz)
            assert np.allclose(b[0:100], time)
            assert c is None
            assert d is None
    
            assert np.allclose(a[100:], xyz)
            assert np.allclose(b[100:], time)
            assert c is None
            assert d is None
    
    def test_write_3(self):
        """Default test using netCDF4"""
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            # you can't supply cell_lengths without cell_angles
            with pytest.raises(ValueError):
                f.write(np.random.randn(100, 3, 3), cell_lengths=np.random.randn(100, 3))
            # or the other way around
            with pytest.raises(ValueError):
                f.write(np.random.randn(100, 3, 3), cell_angles=np.random.randn(100, 3))
    
    def test_n_atoms(self):
        """Default test using netCDF4"""
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(np.random.randn(1, 11, 3))
        with NetCDFTrajectoryFile(temp) as f:
            assert np.allclose(f.n_atoms, 11)
    
    def test_do_overwrite(self):
        """Default test using netCDF4"""
        with open(temp, "w") as f:
            f.write("a")
    
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(np.random.randn(10, 5, 3))
    
    def test_do_not_overwrite(self):
        """Default test using netCDF4"""
        with open(temp, "w") as f:
            f.write("a")
    
        with pytest.raises(IOError):
            with NetCDFTrajectoryFile(temp, "w", force_overwrite=False) as f:
                f.write(np.random.randn(10, 5, 3))
    
    def test_trajectory_save_load(self, pdb_path):
        """Default test using netCDF4"""
        t = md.load(pdb_path)
        t.unitcell_lengths = 1 * np.ones((1, 3))
        t.unitcell_angles = 90 * np.ones((1, 3))
    
        t.save(temp)
        t2 = md.load(temp, top=t.topology)
    
        assert np.allclose(t.xyz, t2.xyz)
        assert np.allclose(t.unitcell_lengths, t2.unitcell_lengths)


class TestNetCDFScipy(TestNetCDFNetCDF4):
    """This inherits the TestNetCDFNetCDF4 class and run all tests with SciPy"""
    def setup_method(self, method):
        """Patching out netCDF4. This is the way to do it inside a class"""
        monkeypatch = MonkeyPatch()
        monkeypatch.setitem(sys.modules, 'netCDF4', None)

    def teardown_method(self, method):
        """Undoing most changes, just in case."""
        monkeypatch = MonkeyPatch()
        monkeypatch.delitem(sys.modules, 'netCDF4', None)

# TODO: Alternative needed here
@needs_cpptraj
def test_cpptraj(get_fn):
    trj0 = md.load(get_fn("frame0.dcd"), top=get_fn("frame0.pdb"))
    trj0.save(temp)

    top = get_fn("frame0.pdb")
    subprocess.check_call(
        [
            "cpptraj",
            "-p",
            top,
            "-y",
            temp,
            "-x",
            temp2,
        ],
    )

    trj1 = md.load(temp, top=top)
    trj2 = md.load(temp2, top=top)

    assert np.allclose(trj0.xyz, trj2.xyz)
    assert np.allclose(trj1.xyz, trj2.xyz)
    assert np.allclose(trj0.unitcell_vectors, trj2.unitcell_vectors)
    assert np.allclose(trj1.unitcell_vectors, trj2.unitcell_vectors,)

    assert np.allclose(trj0.time, trj1.time)
    assert np.allclose(trj0.time, trj2.time)
    assert np.allclose(trj1.time, trj2.time)