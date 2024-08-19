import os
from os import path
from os.path import join
import numpy as np
import pytest
from biotraj import TRRTrajectoryFile
from .util import data_dir


@pytest.fixture(scope="module")
def trr_path():
    return join(data_dir(), "frame0.trr")


@pytest.fixture(scope="module")
def transferred_test_trr():
    return join(data_dir(), "transferred_test_trr.trr")


# Write data and read it
def test_write_reread(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    time = np.random.randn(500)
    step = np.arange(500)
    lambd = np.random.randn(500)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f.write(xyz=xyz, time=time, step=step, lambd=lambd)
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2 = f.read()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# TRR read stride when n_frames is supplied (different path)
def test_read_stride_n_frames(trr_path):
    with TRRTrajectoryFile(trr_path) as f:
        xyz, time, step, box, lambd = f.read()
    with TRRTrajectoryFile(trr_path) as f:
        xyz3, time3, step3, box3, lambd3 = f.read(n_frames=1000, stride=3)
    assert np.allclose(xyz[::3], xyz3)
    assert np.allclose(step[::3], step3)
    assert np.allclose(box[::3], box3)
    assert np.allclose(time[::3], time3)


# Read xtc with stride and offsets
def test_read_stride_offsets(trr_path):
    with TRRTrajectoryFile(trr_path) as f:
        xyz, time, step, box, lambd = f.read()
    for s in (1, 2, 3, 4, 5):
        with TRRTrajectoryFile(trr_path) as f:
            # pre-compute byte offsets between frames
            f.offsets
            xyz_s, time_s, step_s, box_s, lamb_s = f.read(stride=s)
        assert np.allclose(xyz_s, xyz[::s])
        assert np.allclose(step_s, step[::s])
        assert np.allclose(box_s, box[::s])
        assert np.allclose(time_s, time[::s])


# Read trr with stride with n_frames and offsets
def test_read_stride_n_frames_offsets(trr_path):
    with TRRTrajectoryFile(trr_path) as f:
        xyz, time, step, box, lambd = f.read()
    for s in (1, 2, 3, 4, 5):
        with TRRTrajectoryFile(trr_path) as f:
            # pre-compute byte offsets between frames
            f.offsets
            xyz_s, time_s, step_s, box_s, lamb_s = f.read(n_frames=1000, stride=s)
        assert np.allclose(xyz_s, xyz[::s])
        assert np.allclose(step_s, step[::s])
        assert np.allclose(box_s, box[::s])
        assert np.allclose(time_s, time[::s])


# Read TRR with subsequently different strides
def test_read_stride_switching(trr_path):
    with TRRTrajectoryFile(trr_path) as f:
        xyz, time, step, box, lambd = f.read()
    with TRRTrajectoryFile(trr_path) as f:
        # pre-compute byte offsets between frames
        f.offsets
        # read the first 10 frames with stride of 2
        s = 2
        n_frames = 10
        xyz_s, time_s, step_s, box_s, lamb_s = f.read(n_frames=n_frames, stride=s)
        assert np.allclose(xyz_s, xyz[: n_frames * s : s])
        assert np.allclose(step_s, step[: n_frames * s : s])
        assert np.allclose(box_s, box[: n_frames * s : s])
        assert np.allclose(time_s, time[: n_frames * s : s])
        # now read the rest with stride 3, should start from frame index 8.
        # eg. np.arange(0, n_frames*s + 1, 2)[-1] == 18
        offset = f.tell()
        assert offset == 20
        s = 3
        xyz_s, time_s, step_s, box_s, lamb_s = f.read(n_frames=None, stride=s)
        assert np.allclose(xyz_s, xyz[offset::s])
        assert np.allclose(step_s, step[offset::s])
        assert np.allclose(box_s, box[offset::s])
        assert np.allclose(time_s, time[offset::s])


# Write and read with custom file
def test_write_read(transferred_test_trr):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    time = np.random.randn(500)
    step = np.arange(500)
    lambd = np.random.randn(500)

    with TRRTrajectoryFile(transferred_test_trr, "w", force_overwrite=True) as f:
        f.write(xyz=xyz, time=time, step=step, lambd=lambd)
    with TRRTrajectoryFile(transferred_test_trr) as f:
        xyz2, time2, step2, box2, lambd2 = f.read(n_frames=500)
    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# Test a selection of/strided atomindices
def test_read_atomindices_selection(transferred_test_trr):
    # Create auxiliary file
    test_write_read(transferred_test_trr)

    with TRRTrajectoryFile(transferred_test_trr) as f:
        xyz, time, step, box, lambd = f.read()

    with TRRTrajectoryFile(transferred_test_trr) as f:
        xyz2, time2, step2, box2, lambd2 = f.read(atom_indices=[0, 1, 2])
    # Remove auxiliary file
    os.remove(transferred_test_trr)

    assert np.allclose(xyz[:, [0, 1, 2]], xyz2)
    assert np.allclose(step, step2)
    assert np.all([b is None for b in [box, box2]])
    assert np.allclose(lambd, lambd2)
    assert np.allclose(time, time2)


def test_read_atomindices_strided(transferred_test_trr):
    # Create auxiliary file
    test_write_read(transferred_test_trr)

    with TRRTrajectoryFile(transferred_test_trr) as f:
        xyz, time, step, box, lambd = f.read()

    with TRRTrajectoryFile(transferred_test_trr) as f:
        xyz2, time2, step2, box2, lambd2 = f.read(atom_indices=slice(None, None, 2))
    # Remove auxiliary file
    os.remove(transferred_test_trr)
    assert np.allclose(xyz[:, ::2], xyz2)
    assert np.allclose(step, step2)
    assert np.all([b is None for b in [box, box2]])
    assert np.allclose(lambd, lambd2)
    assert np.allclose(time, time2)


# Write data one frame at a time. This checks how the shape is dealt with,
# because each frame is deficient in shape.
def test_deficient_shape(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    time = np.random.randn(500)
    step = np.arange(500)
    lambd = np.random.randn(500)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        for i in range(len(xyz)):
            f.write(xyz=xyz[i], time=time[i], step=step[i], lambd=lambd[i])
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2 = f.read()

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# Test raised error upon addition of box vectors after first writeout
def test_ragged_box_angles_added_second_write_err(tmpdir):
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w", force_overwrite=True) as f:
        f.write(xyz)
        with pytest.raises(ValueError):
            f.write(xyz, time, box)


# Test raised error upon removal of box vectors after first writeout
def test_ragged_box_angles_missing_second_write_err(tmpdir):
    # try first writing no box vectors, and then adding some
    xyz = np.random.randn(100, 5, 3)
    time = np.random.randn(100)
    box = np.random.randn(100, 3, 3)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w", force_overwrite=True) as f:
        f.write(xyz, time=time, box=box)
        with pytest.raises(ValueError):
            f.write(xyz)


# Test, whether writeout of malformed TRRs is correctly aborted
def test_malformed(tmpdir):
    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with open(tmp_file_path, "w") as tmpf:
        # very badly malformed TRR
        tmpf.write("foo")

    with pytest.raises(IOError):
        TRRTrajectoryFile(tmp_file_path)

    psutil = pytest.importorskip("psutil")
    open_files = psutil.Process().open_files()
    paths = [path.realpath(f.path) for f in open_files]
    assert path.realpath(tmp_file_path) not in paths


def test_tell(trr_path):
    with TRRTrajectoryFile(trr_path) as f:
        assert np.allclose(f.tell(), 0)

        f.read(101)
        assert np.allclose(f.tell(), 101)

        f.read(3)
        assert np.allclose(f.tell(), 104)


def test_seek(trr_path):
    reference = TRRTrajectoryFile(trr_path).read()[0]
    with TRRTrajectoryFile(trr_path) as f:
        assert np.allclose(len(f), len(reference))
        assert np.allclose(len(f.offsets), len(reference))

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


## NOTE: The following tests are for a hidden API
def test_get_velocities(tmpdir):
    """Write data with velocities and read it back"""
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    vel = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f._write(xyz=xyz, time=time, step=step, box=box, lambd=lambd, vel=vel)
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2, vel2, forces2 = f._read(
            n_frames=500,
            atom_indices=None,
            get_velocities=True,
            get_forces=False,
        )

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)
    assert np.allclose(vel, vel2)
    assert forces2 is None


# Test: Write data with forces; reread that file
def test_get_forces(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    forces = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f._write(
            xyz=xyz,
            time=time,
            step=step,
            box=box,
            lambd=lambd,
            forces=forces,
        )
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2, vel2, forces2 = f._read(
            n_frames=500,
            atom_indices=None,
            get_velocities=False,
            get_forces=True,
        )

    assert np.allclose(xyz, xyz2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)
    assert vel2 is None
    assert np.allclose(forces, forces2)


# Test: Write data with forces and velocities; reread that file
def test_get_velocities_and_forces(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    vel = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    forces = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f._write(
            xyz=xyz,
            time=time,
            step=step,
            box=box,
            lambd=lambd,
            vel=vel,
            forces=forces,
        )
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2, vel2, forces2 = f._read(
            n_frames=500,
            atom_indices=None,
            get_velocities=True,
            get_forces=True,
        )

    assert np.allclose(xyz, xyz2)
    assert np.allclose(vel, vel2)
    assert np.allclose(forces, forces2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# Test atom indices subsets
def test_get_velocities_forces_atom_indices(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    vel = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    forces = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f._write(
            xyz=xyz,
            time=time,
            step=step,
            box=box,
            lambd=lambd,
            vel=vel,
            forces=forces,
        )
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2, vel2, forces2 = f._read(
            n_frames=500,
            atom_indices=[0, 1, 2],
            get_velocities=True,
            get_forces=True,
        )

    assert np.allclose(xyz[:, [0, 1, 2]], xyz2)
    assert np.allclose(vel[:, [0, 1, 2]], vel2)
    assert np.allclose(forces[:, [0, 1, 2]], forces2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# Test atom indices with strides
def test_get_velocities_forces_atom_indices_strided(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    vel = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    forces = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f._write(
            xyz=xyz,
            time=time,
            step=step,
            box=box,
            lambd=lambd,
            vel=vel,
            forces=forces,
        )
    with TRRTrajectoryFile(tmp_file_path) as f:
        xyz2, time2, step2, box2, lambd2, vel2, forces2 = f._read(
            n_frames=500,
            atom_indices=slice(None, None, 2),
            get_velocities=True,
            get_forces=True,
        )

    assert np.allclose(xyz[:, ::2], xyz2)
    assert np.allclose(vel[:, ::2], vel2)
    assert np.allclose(forces[:, ::2], forces2)
    assert np.allclose(time, time2)
    assert np.allclose(step, step2)
    assert np.allclose(lambd, lambd2)


# Test error upon requesting velocities from a file that lacks these
def test_read_velocities_do_not_exist(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f.write(xyz=xyz, time=time, step=step, box=box, lambd=lambd)
    with TRRTrajectoryFile(tmp_file_path) as f:
        with pytest.raises(RuntimeError):
            f._read(
                n_frames=500,
                atom_indices=None,
                get_velocities=True,
                get_forces=False,
            )


# Test error upon requesting forces from a file that lacks these
def test_read_forces_do_not_exist(tmpdir):
    xyz = np.array(np.random.randn(500, 50, 3), dtype=np.float32)
    box = np.array(np.random.randn(500, 3, 3), dtype=np.float32)
    time = np.array(np.random.randn(500), dtype=np.float32)
    step = np.array(np.arange(500), dtype=np.int32)
    lambd = np.array(np.random.randn(500), dtype=np.float32)

    tmp_file_path = join(tmpdir, "tmpfile.trr")

    with TRRTrajectoryFile(tmp_file_path, "w") as f:
        f.write(xyz=xyz, time=time, step=step, box=box, lambd=lambd)
    with TRRTrajectoryFile(tmp_file_path) as f:
        with pytest.raises(RuntimeError):
            f._read(
                n_frames=500,
                atom_indices=None,
                get_velocities=False,
                get_forces=True,
            )
