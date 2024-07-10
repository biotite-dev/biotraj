from os.path import join
import sys

import numpy as np
import pytest
from biotraj.formats import XTCTrajectoryFile

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

# Test: Read Chunks
def test_read_chunk1(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, "r", chunk_size_multiplier=0.5) as f:
        xyz, time, step, box = f.read()

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])



def test_read_stride(xtc_npz_reference_path, xtc_path, strides):
    # read xtc with stride
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


def test_read_stride_n_frames(xtc_npz_reference_path, xtc_path, strides):
    # read xtc with stride with n_frames
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(n_frames=1000, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


def test_read_stride_offsets(xtc_npz_reference_path, xtc_path, strides):
    # read xtc with stride and offsets
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            f.offsets  # pre-compute byte offsets between frames
            xyz, time, step, box = f.read(stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


def test_read_stride_n_frames_offsets(xtc_npz_reference_path, xtc_path, strides):
    # read xtc with stride with n_frames and offsets
    npz_file = np.load(xtc_npz_reference_path)
    for s in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            f.offsets  # pre-compute byte offsets between frames
            xyz, time, step, box = f.read(n_frames=1000, stride=s)
        assert np.allclose(xyz, npz_file["xyz"][::s])
        assert np.allclose(step, npz_file["step"][::s])
        assert np.allclose(box, npz_file["box"][::s])
        assert np.allclose(time, npz_file["time"][::s])


def test_read_stride_switching_offsets(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        f.offsets  # pre-compute byte offsets between frames
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


def test_read_atomindices_1(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        xyz, time, step, box = f.read(atom_indices=[0, 1, 2])
    assert np.allclose(xyz, npz_file["xyz"][:, [0, 1, 2]])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


def test_read_atomindices_w_stride(xtc_npz_reference_path, xtc_path, strides):
    # test case for bug: https://github.com/mdtraj/mdtraj/issues/1394
    npz_file = np.load(xtc_npz_reference_path)
    for stride in strides:
        with XTCTrajectoryFile(xtc_path) as f:
            xyz, time, step, box = f.read(atom_indices=[0, 1, 2], stride=stride)
        assert np.allclose(xyz, npz_file["xyz"][:, [0, 1, 2]][::stride])
        assert np.allclose(step, npz_file["step"][::stride])
        assert np.allclose(box, npz_file["box"][::stride])
        assert np.allclose(time, npz_file["time"][::stride])


def test_read_atomindices_2(xtc_npz_reference_path, xtc_path):
    npz_file = np.load(xtc_npz_reference_path)
    with XTCTrajectoryFile(xtc_path) as f:
        xyz, time, step, box = f.read(atom_indices=slice(None, None, 2))
    assert np.allclose(xyz, npz_file["xyz"][:, ::2])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


def test_read_chunk2(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, "r", chunk_size_multiplier=1) as f:
        xyz, time, step, box = f.read()

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"])
    assert np.allclose(step, npz_file["step"])
    assert np.allclose(box, npz_file["box"])
    assert np.allclose(time, npz_file["time"])


def test_read_chunk3(xtc_npz_reference_path, xtc_path):
    with XTCTrajectoryFile(xtc_path, chunk_size_multiplier=2) as f:
        xyz, time, step, box = f.read(n_frames=100)

    npz_file = np.load(xtc_npz_reference_path)
    assert np.allclose(xyz, npz_file["xyz"][:100])
    assert np.allclose(step, npz_file["step"][:100])
    assert np.allclose(box, npz_file["box"][:100])
    assert np.allclose(time, npz_file["time"][:100])


def test_write_0(tmpdir, xtc_path):
    with XTCTrajectoryFile(xtc_path) as f:
        xyz = f.read()[0]

    tmpfn = join(tmpdir, "traj.xtc")
    f = XTCTrajectoryFile(tmpfn, "w")
    f.write(xyz)
    f.close()

    with XTCTrajectoryFile(tmpfn) as f:
        xyz2, time2, step2, box2 = f.read()
    assert np.allclose(xyz, xyz2)


def test_write_1(tmpdir):
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


def test_write_2(tmpdir):
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


def test_read_error_0(tmpdir):
    tmpfn = join(tmpdir, "traj.xtc")
    with pytest.raises(IOError):
        with XTCTrajectoryFile(tmpfn, "r") as f:
            f.read()


def test_write_error_0(tmpdir):
    xyz = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        with pytest.raises(ValueError):
            f.read(xyz)


def test_read_error_1():
    with pytest.raises(IOError):
        XTCTrajectoryFile("/tmp/sdfsdfsdf")


def test_read_error_2(dcd_path):
    with pytest.raises(IOError):
        XTCTrajectoryFile(dcd_path).read()


def test_xtc_write_wierd_0(tmpdir):
    x0 = np.asarray(np.random.randn(100, 3, 3), dtype=np.float32)
    x1 = np.asarray(np.random.randn(100, 9, 3), dtype=np.float32)
    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(x0)
        with pytest.raises(ValueError):
            f.write(x1)

    xr = XTCTrajectoryFile(tmpfn).read()[0]
    print(xr.shape)


def test_tell(xtc_path):
    with XTCTrajectoryFile(xtc_path) as f:
        assert np.allclose(f.tell(), 0)

        f.read(101)
        assert np.allclose(f.tell(), 101)

        f.read(3)
        assert np.allclose(f.tell(), 104)


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


def test_seek_out_of_bounds(xtc_path):
    with XTCTrajectoryFile(xtc_path, "r") as fh:
        with pytest.raises(IOError):
            fh.seek(10000000)


def test_ragged_1(tmpdir):
    # try first writing no box vectors,, and then adding some
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w", force_overwrite=True) as f:
        f.write(xyz)
        with pytest.raises(ValueError):
            f.write(xyz, time, box)


def test_ragged_2(tmpdir):
    # try first writing no box vectors, and then adding some
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w", force_overwrite=True) as f:
        f.write(xyz, time=time, box=box)
        with pytest.raises(ValueError):
            f.write(xyz)


def test_short_traj(tmpdir):
    tmpfn = join(tmpdir, "traj.xtc")
    with XTCTrajectoryFile(tmpfn, "w") as f:
        f.write(np.random.uniform(size=(5, 100000, 3)))
    with XTCTrajectoryFile(tmpfn, "r") as f:
        assert len(f) == 5, len(f)


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
