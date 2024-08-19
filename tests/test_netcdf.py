import os
import sys
import tempfile
from os.path import join
from shutil import which
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from biotraj import NetCDFTrajectoryFile
from .util import data_dir


@pytest.fixture(scope="module")
def nc_path():
    return join(data_dir(), "mdcrd.nc")


@pytest.fixture(scope="module")
def pdb_path():
    return join(data_dir(), "native.pdb")


@pytest.fixture(scope="module")
def dcd_frame0_reference_path():
    return join(data_dir(), "frame0.dcd")


@pytest.fixture(scope="module")
def pdb_frame0_reference_path():
    return join(data_dir(), "frame0.pdb")


needs_cpptraj = pytest.mark.skipif(
    which("cpptraj") is None,
    reason=(
        "This test requires cpptraj from AmberTools to be installed "
        "(http://ambermd.org)."
        "Alternatively, a Conda package is also available."
    ),
)

## Create test classes:
## Test with both the netcdf4-python reader as well as the SciPy-Version
fd, temp = tempfile.mkstemp(suffix=".nc")
fd2, temp2 = tempfile.mkstemp(suffix=".nc")


# Default test using netCDF4
class TestNetCDFNetCDF4:
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

    # Test, whether an IOError is raised when reading closed files
    def test_read_after_close(self, nc_path):
        f = NetCDFTrajectoryFile(nc_path)
        assert np.allclose(f.n_atoms, 223)
        assert np.allclose(f.n_frames, 101)

        f.close()

        with pytest.raises(IOError):
            f.read()

    # Test: Read multiple frames in chunks
    def test_read_multi_frame_chunk(self, nc_path):
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

    def test_read_multi_frame_chunk_2(self, nc_path):
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

    # Request large chunk first
    def test_read_multi_frame_chunk_first(self, nc_path):
        with NetCDFTrajectoryFile(nc_path) as file:
            a = file.read(1000000000)
        with NetCDFTrajectoryFile(nc_path) as file:
            b = file.read()

        assert np.allclose(a[0], b[0])

    # Test basic reading and writing
    def test_read_write(self):
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

    # Test raised error upon addition of box vectors after first writeout
    def test_ragged_box_angles_added_second_write_err(self):
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
        cell_lengths = np.random.randn(100, 3)
        cell_angles = np.random.randn(100, 3)

        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time)
            with pytest.raises(ValueError):
                f.write(xyz, time, cell_lengths, cell_angles)

    # Test raised error upon removal of box vectors after first writeout
    def test_ragged_box_angles_missing_second_write_err(self):
        xyz = np.random.randn(100, 3, 3)
        time = np.random.randn(100)
        cell_lengths = np.random.randn(100, 3)
        cell_angles = np.random.randn(100, 3)

        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(xyz, time, cell_lengths, cell_angles)
            with pytest.raises(ValueError):
                f.write(xyz, time)

    def test_read_write_generated_numbers(self):
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

    # Test error: cell_lengths and cell_angles cannot be supplied alone
    def test_write_cell_box_angles(self):
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            # you can't supply cell_lengths without cell_angles
            with pytest.raises(ValueError):
                f.write(
                    np.random.randn(100, 3, 3), cell_lengths=np.random.randn(100, 3)
                )
            # or the other way around
            with pytest.raises(ValueError):
                f.write(np.random.randn(100, 3, 3), cell_angles=np.random.randn(100, 3))

    def test_n_atoms(self):
        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(np.random.randn(1, 11, 3))
        with NetCDFTrajectoryFile(temp) as f:
            assert np.allclose(f.n_atoms, 11)

    def test_do_overwrite(self):
        with open(temp, "w") as f:
            f.write("a")

        with NetCDFTrajectoryFile(temp, "w", force_overwrite=True) as f:
            f.write(np.random.randn(10, 5, 3))

    # Test overwerite behaviour
    def test_do_not_overwrite(self):
        with open(temp, "w") as f:
            f.write("a")

        with pytest.raises(IOError):
            with NetCDFTrajectoryFile(temp, "w", force_overwrite=False) as f:
                f.write(np.random.randn(10, 5, 3))


# Separate class for testing the Scipy fallback-version
class TestNetCDFScipy(TestNetCDFNetCDF4):
    """This inherits the TestNetCDFNetCDF4 class and run all tests with SciPy"""

    def setup_method(self, method):
        """Patching out netCDF4. This is the way to do it inside a class"""
        monkeypatch = MonkeyPatch()
        monkeypatch.setitem(sys.modules, "netCDF4", None)

    def teardown_method(self, method):
        """Undoing most changes, just in case."""
        monkeypatch = MonkeyPatch()
        monkeypatch.delitem(sys.modules, "netCDF4", None)
