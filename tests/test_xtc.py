import sys
from os.path import join
import numpy as np
import pytest
from biotraj import XTCTrajectoryFile
from .util import data_dir


@pytest.fixture(scope="module")
def xtc_path():
    return join(data_dir(), "frame0.xtc")


@pytest.fixture(scope="module")
def xtc_npz_reference_path():
    return join(data_dir(), "frame0.xtc.npz")


@pytest.fixture(scope="module")
def pdb_path():
    return join(data_dir(), "native.pdb")


@pytest.fixture(scope="module")
def dcd_path():
    return join(data_dir(), "frame0.dcd")


@pytest.fixture(scope="module")
def strides():
    return (1, 2, 3, 4, 5, 7, 10, 11)


not_on_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Can not open file being written again due to file locking.",
)


# Test non-default buffer chunk size
def test_read_chunk_buffer_factor_0_5(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, "r", chunk_size_multiplier=0.5) as f:
        xyz, time, step, box = f.read()

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


def test_read_chunk_buffer_factor_1(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, "r", chunk_size_multiplier=1) as f:
        xyz, time, step, box = f.read()

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


def test_read_chunk_buffer_factor_2(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, chunk_size_multiplier=2) as f:
        xyz, time, step, box = f.read(n_frames=100)

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"][:100])
    assert np.allclose(step, npz_file["step"][:100])
    assert np.allclose(box, npz_file["box"][:100])
    assert np.allclose(time, npz_file["time"][:100])


# Read XTC files with strides
def test_read_stride(xtc_npz_reference_path, xtc_path, strides):
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


# Read XTC file strided, for a subset of frames
def test_read_stride_n_frames(xtc_npz_reference_path, xtc_path, strides):
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(n_frames=1000, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


# Read XTC file strided with offsets
def test_read_stride_offsets(xtc_npz_reference_path, xtc_path, strides):
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            # pre-compute byte offsets between frames
            f.offsets
            xyz, time, step, box = f.read(stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


# Read XTC file strided for n_frames with offsets
def test_read_stride_n_frames_offsets(xtc_npz_reference_path, xtc_path, strides):
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            # pre-compute byte offsets between frames
            f.offsets
            xyz, time, step, box = f.read(n_frames=1000, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


# Read XTC file with subsequently different strides
def test_read_stride_switching_offsets(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        # pre-compute byte offsets between frames
        f.offsets
        # read the first 10 frames with stride of 2
        s = 2
        n_frames = 10
        xyz, time, step, box = f.read(n_frames=n_frames, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][: n_frames * s : s])
        assert np.allclose(step, npz_file["step"][: n_frames * s : s])
        assert np.allclose(box, npz_file["box"][: n_frames * s : s])
        assert np.allclose(time, npz_file["time"][: n_frames * s : s])
        # now read the rest with stride 3, should start from frame index 8.
        # eg. np.arange(0, n_frames*s + 1, 2)[-1] == 20
        offset = f.tell()
        assert offset == 20
        s = 3
        xyz, time, step, box = f.read(n_frames=None, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][offset::s])
        assert np.allclose(step, npz_file["step"][offset::s])
        assert np.allclose(box, npz_file["box"][offset::s])
        assert np.allclose(time, npz_file["time"][offset::s])


# Test a selection of atomindices
def test_read_atomindices_selection(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        xyz, time, step, box = f.read(atom_indices=[0, 1, 2])
    assert np.allclose(xyz, npz_file["xyz"][:, [0, 1, 2]])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


# Test a selection of atomindices for strided trajectory frames
# Test case for bug: https://github.com/mdtraj/mdtraj/issues/1394
def test_read_atomindices_selection_strided_frames(
    xtc_npz_reference_path, xtc_path, strides
):
    npz_file = np.load(xtc_npz_reference_path)
    for stride in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(atom_indices=[0, 1, 2], stride=stride)
        assert np.allclose(xyz, npz_file["xyz"][:, [0, 1, 2]][::stride])
        assert np.allclose(step, npz_file["step"][::stride])
        assert np.allclose(box, npz_file["box"][::stride])
        assert np.allclose(time, npz_file["time"][::stride])


# Test strided atomindices
def test_read_atomindices_strided(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        xyz, time, step, box = f.read(atom_indices=slice(None, None, 2))
    assert np.allclose(xyz, npz_file["xyz"][:, ::2])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


# Write coords only/all categories/all categories iteratively
def test_write_coords_only(tmpdir, xtc_path):
    with XTCTrajectoryFile(xtc_path) as f:
        xyz = f.read()[0]

    tmpfn = join(tmpdir, "traj.xtc")
    f = XTCTrajectoryFile(tmpfn, "w")
    f.write(xyz)
    f.close()

    with XTCTrajectoryFile(tmpfn) as f:
        xyz2, time2, step2, box2 = f.read()
    assert np.allclose(xyz, xyz2)


def test_write_coords_time_step_box(tmpdir):
    xyz = np.asarray(np.around(np.random.randn(100, 10, 3), 3), dtype=np.float32)
    time = np.asarray(np.random.randn(100), dtype=np.float32)
    step = np.arange(100)
    box = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(xyz, time=time, step=step, box=box)
    with XTCTrajectoryFile(tmpfn) as f:
        xyz2, time2, step2, box2 = f.read()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(box, box2)


def test_write_coords_time_step_box_iteratively(tmpdir):
    xyz = np.asarray(np.around(np.random.randn(100, 10, 3), 3), dtype=np.float32)
    time = np.asarray(np.random.randn(100), dtype=np.float32)
    step = np.arange(100)
    box = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        for i in range(len(xyz)):
            f.write(xyz[i], time=time[i], step=step[i], box=box[i])
    with XTCTrajectoryFile(tmpfn) as f:
        xyz2, time2, step2, box2 = f.read()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(box, box2)


def test_short_traj(tmpdir):
    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(np.random.uniform(size=(5, 100000, 3)))
    with XTCTrajectoryFile(tmpfn, "r") as f:
        assert len(f) == 5


# Attempt to read an empty file
def test_read_empty_file_error(tmpdir):
    tmpfn = join(tmpdir, "traj.xtc")
    with pytest.raises(IOError):
        with XTCTrajectoryFile(tmpfn, "r") as f:
            f.read()


# Attempt to read from file opened for writing
def test_read_opened_for_write_error(tmpdir):
    xyz = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        with pytest.raises(ValueError):
            f.read(xyz)


# Try to read non-existant file
def test_read_non_existant_error():
    with pytest.raises(IOError):
        XTCTrajectoryFile("/tmp/sdfsdfsdf")


# Attempt to non-XTC file
def test_read_without_opening(dcd_path):
    with pytest.raises(IOError):
        XTCTrajectoryFile(dcd_path).read()


# Attempt subsequent writes with deformed array
def test_xtc_write_weird(tmpdir):
    x0 = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)
    x1 = np.asarray(np.random.randn(100, 9, 3), dtype=np.float32)
    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(x0)
        with pytest.raises(ValueError):
            f.write(x1)


# Test tell function -> current position in file
def test_tell(xtc_path):
    with XTCTrajectoryFile(xtc_path) as f:
        assert np.allclose(f.tell(), 0)

        f.read(101)
        assert np.allclose(f.tell(), 101)

        f.read(3)
        assert np.allclose(f.tell(), 104)


# Test seek function -> Move to a new position in file
def test_seek(xtc_path):
    reference = XTCTrajectoryFile(xtc_path).read()[0]
    with XTCTrajectoryFile(xtc_path) as f:
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

        f.seek(5)  # offset array is going to be built
        assert len(f.offsets) == len(reference)
        assert np.allclose(f.read(1)[0][0], reference[5])
        assert np.allclose(f.tell(), 6)

        f.seek(-5, 1)
        assert np.allclose(f.tell(), 1)
        assert np.allclose(f.read(1)[0][0], reference[1])


# Test seek with subset of atoms
def test_seek_natoms9(tmpdir, xtc_path):
    # create a xtc file with 9 atoms and seek it.
    with XTCTrajectoryFile(xtc_path, "r") as fh:
        xyz = fh.read()[0][:, :9, :]

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w", force_overwrite=True) as f:
        f.write(xyz)

    with XTCTrajectoryFile(tmpfn, "r") as f:
        assert np.allclose(f.read(1)[0].shape, (1, 9, 3))
        assert np.allclose(f.tell(), 1)
        f.seek(99)
        assert np.allclose(f.read(1)[0].squeeze(), xyz[99])
        # seek relative
        f.seek(-1, 1)
        assert np.allclose(f.read(1)[0].squeeze(), xyz[99])

        f.seek(0, 0)
        assert np.allclose(f.read(1)[0].squeeze(), xyz[0])


# Test, whether error is raised after attempted out-of-bounds seek
def test_seek_out_of_bounds(xtc_path):
    with XTCTrajectoryFile(xtc_path, "r") as fh:
        with pytest.raises(IOError):
            fh.seek(10000000)


# Test raised error upon addition of box vectors after first writeout
def test_ragged_box_angles_added_second_write_err(tmpdir):
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w", force_overwrite=True) as f:
        f.write(xyz)
        with pytest.raises(ValueError):
            f.write(xyz, time, box)


# Test raised error upon removal of box vectors after first writeout
def test_ragged_box_angles_missing_second_write_err(tmpdir):
    # try first writing no box vectors, and then adding some
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w", force_overwrite=True) as f:
        f.write(xyz, time=time, box=box)
        with pytest.raises(ValueError):
            f.write(xyz)


# Write to file immediately, keep file open for rereading
@not_on_win
def test_flush(tmpdir):
    tmpfn = join(tmpdir, "traj.xtc")
    data = np.random.random((5, 100, 3))
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(data)
        f.flush()
        # note that f is still open, so we can now try to read the contents flushed to disk.
        with XTCTrajectoryFile(tmpfn, "r") as f2:
            out = f2.read()
        assert np.allclose(out[0], data, atol=1e-3)
