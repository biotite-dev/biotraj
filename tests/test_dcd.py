from os.path import join
import numpy as np
import pytest
from biotraj import DCDTrajectoryFile
from .util import data_dir


@pytest.fixture(scope="module")
def dcd_path():
    return join(data_dir(), "frame0.dcd")


@pytest.fixture(scope="module")
def dcd_npz_reference_path():
    return join(data_dir(), "frame0.dcd.npz")


## Trajectory read-in
# Compare DCD reader results with NPZ coordinate reference
def test_read(dcd_path, dcd_npz_reference_path):
    xyz, box_lengths, box_angles = DCDTrajectoryFile(dcd_path).read()
    xyz2 = np.load(dcd_npz_reference_path)["dcd_coords"]

    assert np.allclose(xyz, xyz2)


# Compare alternative modes: Full compared to N frames
def test_read_n_frames(dcd_path):
    xyz1, box_lengths1, box_angles1 = DCDTrajectoryFile(dcd_path).read()
    xyz2, box_lengths2, box_angles2 = DCDTrajectoryFile(dcd_path).read(10000)

    assert np.allclose(xyz1, xyz2)
    assert np.allclose(box_lengths1, box_lengths2)
    assert np.allclose(box_angles1, box_angles2)


# Read DCD file with strides
def test_read_stride(dcd_path):
    with DCDTrajectoryFile(dcd_path) as f:
        xyz1, box_lengths1, box_angles1 = f.read()
    with DCDTrajectoryFile(dcd_path) as f:
        xyz2, box_lengths2, box_angles2 = f.read(stride=2)

    assert np.allclose(xyz1[::2], xyz2)
    assert np.allclose(box_lengths1[::2], box_lengths2)
    assert np.allclose(box_angles1[::2], box_angles2)


# Read DCD file strided, for a subset of frames
def test_read_stride_n_frame_subset(dcd_path):
    with DCDTrajectoryFile(dcd_path) as f:
        xyz1, box_lengths1, box_angles1 = f.read()
    with DCDTrajectoryFile(dcd_path) as f:
        xyz2, box_lengths2, box_angles2 = f.read(n_frames=1000, stride=2)

    assert np.allclose(xyz1[::2], xyz2)
    assert np.allclose(box_lengths1[::2], box_lengths2)
    assert np.allclose(box_angles1[::2], box_angles2)


# Test iterative read-in with one frame at a time/streaming read
def test_iterative_read(dcd_path):
    xyz_ref, box_lengths_ref, box_angles_ref = DCDTrajectoryFile(dcd_path).read()

    reader = DCDTrajectoryFile(dcd_path)
    for i in range(len(xyz_ref)):
        xyz, box_lenths, box_angles = reader.read(1)
        assert np.allclose(xyz_ref[np.newaxis, i], xyz)
        assert np.allclose(box_lengths_ref[np.newaxis, i], box_lenths)
        assert np.allclose(box_angles_ref[np.newaxis, i], box_angles)


# Test partial streaming read: Streaming followed by remainder of trajectory
def test_partial_streaming(dcd_path):
    xyz_ref, box_lengths_ref, box_angles_ref = DCDTrajectoryFile(dcd_path).read()

    reader = DCDTrajectoryFile(dcd_path)
    for i in range(len(xyz_ref) // 2):
        xyz, box_lenths, box_angles = reader.read(1)
        assert np.allclose(xyz_ref[np.newaxis, i], xyz)
        assert np.allclose(box_lengths_ref[np.newaxis, i], box_lenths)
        assert np.allclose(box_angles_ref[np.newaxis, i], box_angles)

    xyz_rest, box_rest, angles_rest = reader.read()
    i = len(xyz_ref) // 2
    assert np.allclose(xyz_ref[i:], xyz_rest)
    assert np.allclose(box_lengths_ref[i:], box_rest)
    assert np.allclose(box_angles_ref[i:], angles_rest)

    assert len(xyz_ref) == i + len(xyz_rest)


# Test loading subset of atoms
def test_subset_atom_read_in(dcd_path):
    with DCDTrajectoryFile(dcd_path) as f:
        xyz_ref, box_lengths_ref, box_angles_ref = f.read()
    with DCDTrajectoryFile(dcd_path) as f:
        xyz, box_lengths, box_angles = f.read(atom_indices=[1, 2, 5])

    assert np.allclose(xyz_ref[:, [1, 2, 5], :], xyz)


# Sliced atom indices
def test_sliced_atom_read_in(dcd_path):
    with DCDTrajectoryFile(dcd_path) as f:
        xyz_ref, box_lengths_ref, box_angles_ref = f.read()
    with DCDTrajectoryFile(dcd_path) as f:
        xyz, box_lengths, box_angles = f.read(atom_indices=slice(None, None, 2))

    assert np.allclose(xyz_ref[:, ::2, :], xyz)


## Write
# Test basic writeout, single line
def test_write_single_line(tmpdir, dcd_path):
    fn = join(tmpdir, "x.dcd")
    with DCDTrajectoryFile(dcd_path) as f:
        xyz = f.read()[0]
    with DCDTrajectoryFile(fn, "w") as f:
        f.write(xyz)
    with DCDTrajectoryFile(fn) as f:
        xyz2 = f.read()[0]

    assert np.allclose(xyz, xyz2)


# Box length and angles
def test_write_with_box(tmpdir):
    fn = join(tmpdir, "x.dcd")
    xyz = np.array(np.random.randn(500, 10, 3), dtype=np.float32)
    box_lengths = 25 * np.ones((500, 3), dtype=np.float32)
    box_angles = 90 * np.ones((500, 3), dtype=np.float32)
    box_lengths[0, 0] = 10.0

    f = DCDTrajectoryFile(fn, "w")
    f.write(xyz, box_lengths, box_angles)
    f.close()

    f = DCDTrajectoryFile(fn)
    xyz2, box_lengths2, box_angles2 = f.read()
    f.close()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(box_lengths, box_lengths2)
    assert np.allclose(box_angles, box_angles2)


# Test, whether ValueError is correctly raised for mismatching input
def test_coordinates_boxlength_mismatch(tmpdir):
    fn = join(tmpdir, "x.dcd")
    xyz = np.array(np.random.randn(500, 10, 3), dtype=np.float32)
    box_lengths = 25 * np.ones((600, 3), dtype=np.float32)

    with DCDTrajectoryFile(fn, "w") as f:
        with pytest.raises(ValueError):
            f.write(xyz, box_lengths)


# Test iterative trajectory writing
def test_write_iteratively(tmpdir):
    fn = join(tmpdir, "x.dcd")
    xyz = np.array(np.random.randn(500, 10, 3), dtype=np.float32)
    box_lengths = 25 * np.ones((500, 3), dtype=np.float32)
    box_angles = 90 * np.ones((500, 3), dtype=np.float32)
    box_lengths[0, 0] = 10.0

    f = DCDTrajectoryFile(fn, "w")
    for i in range(len(xyz)):
        f.write(xyz[i], box_lengths[i], box_angles[i])
    f.close()

    f = DCDTrajectoryFile(fn)
    xyz2, box_lengths2, box_angles2 = f.read()
    f.close()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(box_lengths, box_lengths2)
    assert np.allclose(box_angles, box_angles2)


# Test overwriting behaviour and raised Errors
# TODO: TypeCastPerformanceWarning: Casting xyz dtype=float64 to <class 'numpy.float32'>
def test_do_overwrite(tmpdir):
    fn = join(tmpdir, "x.dcd")
    with open(fn, "w") as f:
        f.write("a")

    with DCDTrajectoryFile(fn, "w", force_overwrite=True) as f:
        f.write(np.random.randn(10, 5, 3))


def test_dont_overwrite(tmpdir):
    fn = join(tmpdir, "x.dcd")
    with open(fn, "w") as f:
        f.write("a")

    with pytest.raises(IOError):
        with DCDTrajectoryFile(fn, "w", force_overwrite=False) as f:
            f.write(np.random.randn(10, 5, 3))


# Test reading/writing closed trajs -> IOError
def test_read_closed(dcd_path):
    fn_dcd = dcd_path
    with pytest.raises(IOError):
        f = DCDTrajectoryFile(fn_dcd)
        f.close()
        f.read()


def test_write_closed(dcd_path):
    fn_dcd = dcd_path
    with pytest.raises(IOError):
        f = DCDTrajectoryFile(fn_dcd, "w")
        f.close()
        f.write(np.random.randn(10, 3, 3))


#  Test tell function -> Returns the current file position
def test_tell(dcd_path):
    with DCDTrajectoryFile(dcd_path) as f:
        assert np.allclose(f.tell(), 0)

        f.read(101)
        assert np.allclose(f.tell(), 101)

        f.read(3)
        assert np.allclose(f.tell(), 104)


# Test seek function -> changes the current trajectory file position
def test_seek(dcd_path):
    reference = DCDTrajectoryFile(dcd_path).read()[0]
    with DCDTrajectoryFile(dcd_path) as f:
        assert np.allclose(f.tell(), 0)
        assert np.allclose(f.read(1)[0][0], reference[0])
        assert np.allclose(f.tell(), 1)

        xyz = f.read(1)[0][0]
        assert np.allclose(xyz, reference[1])
        assert np.allclose(f.tell(), 2)

        f.seek(0)
        assert np.allclose(f.tell(), 0)
        xyz = f.read(1)[0][0]
        assert np.allclose(f.tell(), 1)
        assert np.allclose(xyz, reference[0])

        f.seek(5)
        assert np.allclose(f.read(1)[0][0], reference[5])
        assert np.allclose(f.tell(), 6)

        f.seek(-5, 1)
        assert np.allclose(f.tell(), 1)
        assert np.allclose(f.read(1)[0][0], reference[1])


# Test reaised error for delayed addition of cell angles/length
def test_delayed_cell_length_angles(tmpdir):
    fn = join(tmpdir, "x.dcd")
    xyz = np.random.randn(100, 5, 3)
    cell_lengths = np.random.randn(100, 3)
    cell_angles = np.random.randn(100, 3)

    with DCDTrajectoryFile(fn, "w", force_overwrite=True) as f:
        f.write(xyz)
        with pytest.raises(ValueError):
            f.write(xyz, cell_lengths, cell_angles)


def test_delayed_cell_length_angles_2(tmpdir):
    fn = join(tmpdir, "x.dcd")
    xyz = np.random.randn(100, 5, 3)
    cell_lengths = np.random.randn(100, 3)
    cell_angles = np.random.randn(100, 3)

    with DCDTrajectoryFile(fn, "w", force_overwrite=True) as f:
        f.write(xyz, cell_lengths, cell_angles)
        with pytest.raises(ValueError):
            f.write(xyz)
