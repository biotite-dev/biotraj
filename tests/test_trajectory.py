import functools
import sys
from collections import namedtuple
from os.path import join
from pathlib import Path

import numpy as np
import pytest

import biotraj as md
import biotraj.core.trajectory
import biotraj.formats
import biotraj.utils

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
@pytest.fixture(params=file_objs, ids=lambda x: x[1])
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
    return fext not in ["dcd", "pdb", "pdb.gz"]


# "xyz", "lammpstrj", "lh5" -> 3, 6 otherwise
# NOTE: Keep in case xyz/lammpstrj are readded again
def precision(fext):
    # if fext in ["xyz", "lammpstrj", "lh5"]:
    #    return 1e-03
    # else:
    #    return 1e-06
    return 1e-06


def precision2(fext1, fext2):
    return min(precision(fext1), precision(fext2))


# Load trajectory with larger topology -> Test raised error
# (22 atoms in XTC v. 2_000 atoms)
def test_mismatch(xtc_path, larger_topol_path=data_path("4ZUO.pdb")):
    with pytest.raises(ValueError):
        md.load(xtc_path, top=larger_topol_path)


# Test correct handling of boxes after manual addition to PDB w/o box
def test_box(pdb_path=data_path("native.pdb")):
    t = md.load(pdb_path)
    assert t.unitcell_vectors is None
    assert t.unitcell_lengths is None
    assert t.unitcell_angles is None
    assert t.unitcell_volumes is None
    t.unitcell_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3)
    assert np.allclose(np.array([1.0, 1.0, 1.0]), t.unitcell_lengths[0])
    assert np.allclose(np.array([90.0, 90.0, 90.0]), t.unitcell_angles[0])
    assert np.allclose(np.array([1.0]), t.unitcell_volumes)


# Test, whether small boxes are read as is for PDB files
# (unreasonably small boxes are removed heuristically with standard settings).
def test_load_pdb_box(pdb_path=data_path("native2.pdb")):
    t = md.load(pdb_path, no_boxchk=True)
    assert np.allclose(t.unitcell_lengths[0], np.array([0.1, 0.2, 0.3]))
    assert np.allclose(t.unitcell_angles[0], np.array([90.0, 90.0, 90.0]))
    assert np.allclose(
        t.unitcell_vectors[0], np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    )


# Check whether box information is preserved over load/save
# (PDB -> Traj formats)
def test_box_load_save(
    write_traj_with_box,
    pdb_path=data_path("native.pdb"),
    pdb_path_2=data_path("native2.pdb"),
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
    intraj_prep_top=data_path("traj_prep_top.pdb"),
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
        (
            t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)
        ).time,
        t.slice(key=range(10), copy=False).time,
    )
    assert np.allclose(
        (
            t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)
        ).unitcell_vectors,
        t.slice(key=range(10), copy=False).unitcell_vectors,
    )
    assert np.allclose(
        (
            t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)
        ).unitcell_lengths,
        t.slice(key=range(10), copy=False).unitcell_lengths,
    )
    assert np.allclose(
        (
            t.slice(key=range(5), copy=False) + t.slice(key=range(5, 10), copy=False)
        ).unitcell_angles,
        t.slice(key=range(10), copy=False).unitcell_angles,
    )


def test_slice_2(
    intraj_prep=data_path("traj_prep.xtc"),
    intraj_prep_top=data_path("traj_prep_top.pdb"),
):
    t = md.load(intraj_prep, top=intraj_prep_top)

    # with copying
    assert t[0] == t[[0, 1]][0]
    # without copying (in place)
    assert t.slice(key=0, copy=False) == t.slice(key=[0, 1], copy=True)[0]


def test_read_path(ref_traj, monkeypatch):
    if ref_traj.fext in ("nc"):
        # Running with scipy instead of netcdf4
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            md.load(data_path(ref_traj.fn), top=data_path("native.pdb"))

    md.load(data_path(ref_traj.fn), top=data_path("native.pdb"))


def test_write_path(write_traj, monkeypatch):
    # NOTE: Relevant, once topology formats are radded
    if write_traj.fext in ("ncrst", "rst7"):
        pytest.skip(f"{write_traj.fext} can only store 1 frame per file")
    if write_traj.fext in ("mdcrd"):
        pytest.skip(f"{write_traj.fext} can only store rectilinear boxes")

    def test_base(write_traj):
        t = md.load(data_path("traj_prep_top.pdb"))
        if t.unitcell_vectors is None:
            # NOTE: Relevant, once DTR and LAMMPstrj are readded
            if write_traj.fext in ("dtr", "lammpstrj"):
                pytest.skip(f"{write_traj.fext} needs to write unitcells")
        t.save(Path(write_traj.fn))

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_read_write(ref_traj, write_traj, monkeypatch):
    if write_traj.fext in ("ncrst", "rst7"):
        pytest.skip(f"{write_traj.fext} can only store 1 frame per file")
    if write_traj.fext in ("mdcrd"):
        pytest.skip(f"{write_traj.fext} can only store rectilinear boxes")

    def test_base(ref_traj, write_traj):
        top = data_path("native.pdb")
        t = md.load(data_path(ref_traj.fn), top=top)

        if t.unitcell_vectors is None:
            if write_traj.fext in ("dtr", "lammpstrj"):
                pytest.skip(f"{write_traj.fext} needs to write unitcells")

        t.save(write_traj.fn)
        t2 = md.load(write_traj.fn, top=top)
        assert np.allclose(
            t.xyz, t2.xyz, atol=precision2(ref_traj.fext, write_traj.fext)
        )
        if has_time_info(write_traj.fext):
            assert np.allclose(t.time, t2.time, atol=1e-03)

    if write_traj.fext in ("nc") or ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj, write_traj)

    test_base(ref_traj, write_traj)


def test_load(ref_traj, monkeypatch):
    def test_base(ref_traj):
        nat = md.load(data_path("native.pdb"))
        num_block = 3
        t0 = md.load(data_path(ref_traj.fn), top=nat, discard_overlapping_frames=True)
        t1 = md.load(data_path(ref_traj.fn), top=nat, discard_overlapping_frames=False)
        t2 = md.load(
            [data_path(ref_traj.fn) for _ in range(num_block)],
            top=nat,
            discard_overlapping_frames=False,
        )
        t3 = md.load(
            [data_path(ref_traj.fn) for _ in range(num_block)],
            top=nat,
            discard_overlapping_frames=True,
        )

        # these don't actually overlap, so discard_overlapping_frames should
        # have no effect. the overlap is between the last frame of one and the
        # first frame of the next.
        assert np.allclose(t0.n_frames, t1.n_frames)
        assert np.allclose(t0.n_frames * num_block, t2.n_frames)
        assert np.allclose(t3.n_frames, t2.n_frames)

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_float_atom_indices_exception(ref_traj, monkeypatch):
    def test_base(ref_traj):
        # Is an informative error message given when you supply floats for atom_indices?
        top = md.load(data_path("native.pdb")).topology

        try:
            md.load(data_path(ref_traj.fn), atom_indices=[0.5, 1.3], top=top)
        except ValueError as e:
            assert (
                e.args[0]
                == "indices must be of an integer type. float64 is not an integer type"
            )

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_restrict_atoms():
    traj = md.load(data_path("traj_prep.xtc"), top=data_path("traj_prep_top.pdb"))
    time_address = traj.time.ctypes.data

    desired_atom_indices = [0, 1, 2, 5]
    traj.restrict_atoms(desired_atom_indices)
    atom_indices = [a.index for a in traj.top.atoms]
    assert np.allclose([0, 1, 2, 3], atom_indices)
    assert np.allclose(traj.xyz.shape[1], 4)
    assert np.allclose(traj.n_atoms, 4)
    assert np.allclose(traj.n_residues, 1)
    assert np.allclose(len(traj.top._bonds), 2)
    assert np.allclose(traj.n_residues, traj.topology._numResidues)
    assert np.allclose(traj.n_atoms, traj.topology._numAtoms)
    assert np.allclose(
        np.array([a.index for a in traj.topology.atoms]), np.arange(traj.n_atoms)
    )

    # assert that the time field was not copied
    assert traj.time.ctypes.data == time_address


def test_restrict_atoms_not_inplace():
    traj = md.load(data_path("traj_prep.xtc"), top=data_path("traj_prep_top.pdb"))
    traj_backup = md.load(
        data_path("traj_prep.xtc"), top=data_path("traj_prep_top.pdb")
    )
    desired_atom_indices = [0, 1, 2, 5]

    sliced = traj.restrict_atoms(desired_atom_indices, inplace=False)

    # make sure the original one was not modified
    assert np.allclose(traj.xyz, traj_backup.xyz)
    assert np.all(
        [i == j for i, j in zip(traj.topology.atoms, traj_backup.topology.atoms)]
    )

    assert np.allclose(list(range(4)), [a.index for a in sliced.top.atoms])
    assert np.allclose(sliced.xyz.shape[1], 4)
    assert np.allclose(sliced.n_atoms, 4)
    assert np.allclose(sliced.n_residues, 1)
    assert np.allclose(len(sliced.top._bonds), 2)
    assert np.allclose(sliced.n_residues, sliced.topology._numResidues)
    assert np.allclose(sliced.n_atoms, sliced.topology._numAtoms)
    assert np.allclose(
        np.array([a.index for a in sliced.topology.atoms]), np.arange(sliced.n_atoms)
    )

    # make sure the two don't alias the same memory
    assert traj.time.ctypes.data != sliced.time.ctypes.data
    assert traj.unitcell_angles.ctypes.data != sliced.unitcell_angles.ctypes.data
    assert traj.unitcell_lengths.ctypes.data != sliced.unitcell_lengths.ctypes.data


def test_array_vs_matrix():
    top = md.load(data_path("native.pdb")).topology
    xyz = np.random.randn(1, 22, 3)
    xyz_mat = np.matrix(xyz)
    t1 = md.Trajectory(xyz, top)
    t2 = md.Trajectory(xyz_mat, top)

    assert np.allclose(t1.xyz, xyz)
    assert np.allclose(t2.xyz, xyz)


def test_pdb_unitcell_loadsave(tmpdir):
    # Make sure that nonstandard unitcell dimensions are saved and loaded
    # correctly with PDB
    tref = md.load(data_path("native.pdb"))
    tref.unitcell_lengths = 1 + 0.1 * np.random.randn(tref.n_frames, 3)
    tref.unitcell_angles = 90 + 0.0 * np.random.randn(tref.n_frames, 3)
    fn = f"{tmpdir}/x.pdb"
    tref.save(fn)

    tnew = md.load(fn)
    assert np.allclose(tref.unitcell_vectors, tnew.unitcell_vectors, atol=1e-03)


def test_load_combination(ref_traj, monkeypatch):
    # Test that the load function's stride and atom_indices work across
    # all trajectory formats

    def test_base(ref_traj):
        topology = md.load(data_path("native.pdb")).topology
        ainds = np.array([a.index for a in topology.atoms if a.element.symbol == "C"])

        no_kwargs = md.load(data_path(ref_traj.fn), top=topology)
        strided3 = md.load(data_path(ref_traj.fn), top=topology, stride=3)
        subset = md.load(data_path(ref_traj.fn), top=topology, atom_indices=ainds)

        # test 1
        t1 = no_kwargs
        t2 = strided3
        assert np.allclose(t1.xyz[::3], t2.xyz)
        assert np.allclose(t1.time[::3], t2.time)
        if t1.unitcell_vectors is not None:
            assert np.allclose(t1.unitcell_vectors[::3], t2.unitcell_vectors)
        assert np.all([i == j for i, j in zip(t1.topology.atoms, t2.topology.atoms)])

        # test 2
        t1 = no_kwargs
        t2 = subset
        assert np.allclose(t1.xyz[:, ainds, :], t2.xyz)
        assert np.allclose(t1.time, t2.time)
        if t1.unitcell_vectors is not None:
            assert np.allclose(t1.unitcell_vectors, t2.unitcell_vectors)
        assert np.all(
            [i == j for i, j in zip(t1.topology.subset(ainds).atoms, t2.topology.atoms)]
        )

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_no_topology():
    # We can make trajectories without a topology
    md.Trajectory(xyz=np.random.randn(10, 5, 3), topology=None)


def test_join():
    xyz = np.random.rand(10, 5, 3)
    # overlapping frames
    t1 = md.Trajectory(xyz=xyz[:5], topology=None)
    t2 = md.Trajectory(xyz=xyz[4:], topology=None)

    t3 = t1.join(t2, discard_overlapping_frames=True)
    t4 = t1.join(t2, discard_overlapping_frames=False)
    assert np.allclose(t3.xyz, xyz)
    assert np.allclose(len(t4.xyz), 11)
    assert np.allclose(t4.xyz, np.vstack((xyz[:5], xyz[4:])))


def test_md_join():
    fn = data_path("traj_prep.xtc")
    t_ref = md.load(data_path("frame0.xtc"), top=data_path("traj_prep_top.pdb"))[:20]
    loaded = md.load(fn, top=t_ref, stride=2)
    iterloaded = md.join(md.iterload(fn, top=t_ref, stride=2, chunk=6))
    assert np.allclose(loaded.xyz, iterloaded.xyz)
    assert np.allclose(loaded.time, iterloaded.time)
    assert np.allclose(loaded.unitcell_angles, iterloaded.unitcell_angles)
    assert np.allclose(loaded.unitcell_lengths, iterloaded.unitcell_lengths)


def test_stack_1():
    t1 = md.load(data_path("native.pdb"))
    t2 = t1.stack(t1)
    assert np.allclose(t2.n_atoms, 2 * t1.n_atoms)
    assert np.allclose(t2.topology._numAtoms, 2 * t1.n_atoms)
    assert np.allclose(t1.xyz, t2.xyz[:, 0 : t1.n_atoms])
    assert np.allclose(t1.xyz, t2.xyz[:, t1.n_atoms :])


def test_stack_2():
    t1 = md.Trajectory(xyz=np.random.rand(10, 5, 3), topology=None)
    t2 = md.Trajectory(xyz=np.random.rand(10, 6, 3), topology=None)
    t3 = t1.stack(t2)

    assert np.allclose(t3.xyz[:, :5], t1.xyz)
    assert np.allclose(t3.xyz[:, 5:], t2.xyz)
    assert np.allclose(t3.n_atoms, 11)


def test_seek_read_mode(ref_traj, monkeypatch):
    # Test the seek/tell capacity of the different TrajectoryFile objects in
    # read mode. Basically, we just seek around the files and read different
    # segments, keeping track of our location manually and checking with both
    # tell() and by checking that the right coordinates are actually returned
    fobj = ref_traj.fobj
    fn = ref_traj.fn

    if ref_traj.fobj is md.formats.PDBTrajectoryFile:
        pytest.xfail("PDB Files don't support seeking")
    if ref_traj.fext == "xyz.gz":
        pytest.xfail("This is broken")
    if ref_traj.fext == "gro":
        pytest.xfail("This is broken")

    def test_base(ref_traj):
        point = 0
        xyz = md.load(data_path(fn), top=data_path("native.pdb")).xyz
        length = len(xyz)

        with fobj(data_path(fn)) as f:
            for i in range(100):
                r = np.random.rand()
                if r < 0.25:
                    offset = np.random.randint(-5, 5)
                    if 0 < point + offset < length:
                        point += offset
                        f.seek(offset, 1)
                    else:
                        f.seek(0)
                        point = 0
                if r < 0.5:
                    offset = np.random.randint(1, 10)
                    if point + offset < length:
                        read = f.read(offset)
                        # NOTE: Relevant, once XYZ is readded
                        # if fobj not in [
                        #    md.formats.LH5TrajectoryFile,
                        #    md.formats.XYZTrajectoryFile,
                        # ]:
                        #    read = read[0]
                        read = read[0]
                        readlength = len(read)
                        read = biotraj.utils.in_units_of(
                            read, f.distance_unit, "nanometers"
                        )
                        assert np.allclose(xyz[point : point + offset], read)
                        point += readlength
                elif r < 0.75:
                    offset = np.random.randint(low=-100, high=0)
                    try:
                        f.seek(offset, 2)
                        point = length + offset
                    except NotImplementedError:
                        # not all of the *TrajectoryFiles currently support
                        # seeking from the end, so we'll let this pass if they
                        # say that they dont implement this.
                        pass
                else:
                    offset = np.random.randint(100)
                    f.seek(offset, 0)
                    point = offset

                assert np.allclose(f.tell(), point)

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_load_frame(ref_traj, monkeypatch):
    if ref_traj.fobj is md.formats.GroTrajectoryFile:
        pytest.xfail("Gro doesn't implement seek")

    def test_base(ref_traj):
        trajectory = md.load(data_path(ref_traj.fn), top=data_path("native.pdb"))
        rand = np.random.randint(len(trajectory))
        frame = md.load_frame(
            data_path(ref_traj.fn), index=rand, top=data_path("native.pdb")
        )

        assert np.allclose(trajectory[rand].xyz, frame.xyz)
        assert np.allclose(trajectory[rand].unitcell_vectors, frame.unitcell_vectors)
        if has_time_info(ref_traj.fext):
            assert np.allclose(trajectory[rand].time, frame.time)

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_load_frame_2eqq():
    t1 = md.load(data_path("2EQQ.pdb"))
    r = np.random.randint(len(t1))
    t2 = md.load_frame(data_path("2EQQ.pdb"), r)
    assert np.allclose(t1[r].xyz, t2.xyz)


def test_iterload(write_traj, monkeypatch):
    if write_traj.fext == "dtr":
        pytest.xfail("This is broken with dtr")

    def test_base(write_traj):
        t_ref = md.load(data_path("frame0.xtc"), top=data_path("frame0.pdb"))[:20]

        if write_traj.fext in ("ncrst", "rst7"):
            pytest.skip("Only 1 frame per file format")

        t_ref.save(write_traj.fn)

        for stride in [1, 2, 3]:
            loaded = md.load(write_traj.fn, top=t_ref, stride=stride)
            iterloaded = functools.reduce(
                lambda a, b: a.join(b),
                md.iterload(write_traj.fn, top=t_ref, stride=stride, chunk=6),
            )
            assert np.allclose(loaded.xyz, iterloaded.xyz)
            assert np.allclose(loaded.time, iterloaded.time)
            assert np.allclose(loaded.unitcell_angles, iterloaded.unitcell_angles)
            assert np.allclose(loaded.unitcell_lengths, iterloaded.unitcell_lengths)

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_iterload_skip(ref_traj, monkeypatch):
    if ref_traj.fobj is md.formats.PDBTrajectoryFile:
        pytest.xfail("PDB Iterloads an extra frame!!")
    if ref_traj.fobj is md.formats.GroTrajectoryFile:
        pytest.xfail("Not implemented for some reason")
    if ref_traj.fext in ("ncrst", "rst7"):
        pytest.skip("Only 1 frame per file format")

    def test_base(ref_traj):
        top = md.load(data_path("native.pdb"))
        t_ref = md.load(data_path(ref_traj.fn), top=top)

        for cs in [0, 1, 11, 100]:
            for skip in [0, 1, 20, 101]:
                t = functools.reduce(
                    lambda a, b: a.join(b),
                    md.iterload(data_path(ref_traj.fn), skip=skip, top=top, chunk=cs),
                )
                assert np.allclose(t_ref.xyz[skip:], t.xyz)
                assert np.allclose(t_ref.time[skip:], t.time)
                assert np.all(
                    [i == j for i, j in zip(t_ref.topology.atoms, t.topology.atoms)]
                )

    if ref_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(ref_traj)

    test_base(ref_traj)


def test_iterload_chunk_dcd():
    # Makes sure that the actual chunk size yielded by iterload corresponds to the number of
    # frames specified when calling it (for dcd files).
    file = data_path("alanine-dipeptide-explicit.dcd")
    top = data_path("alanine-dipeptide-explicit.pdb")

    skip_frames = 3
    frames_chunk = 2

    full = md.load(file, top=top, stride=skip_frames)

    chunks = []
    for traj_chunk in md.iterload(
        file,
        top=top,
        stride=skip_frames,
        chunk=frames_chunk,
    ):
        chunks.append(traj_chunk)
    joined = md.join(chunks)
    assert len(full) == len(joined)
    assert np.allclose(full.xyz, joined.xyz)


def test_save_load(write_traj, monkeypatch):
    # this cycles all the known formats you can save to, and then tries
    # to reload, using just a single-frame file.

    def test_base(write_traj):
        t_ref = md.load(data_path("native.pdb"))
        t_ref.unitcell_vectors = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        t_ref.save(write_traj.fn)
        t = md.load(write_traj.fn, top=t_ref.topology)

        assert np.allclose(t.xyz, t_ref.xyz)
        assert np.allclose(t.time, t_ref.time)
        if t._have_unitcell:
            assert np.allclose(t.unitcell_angles, t_ref.unitcell_angles)
            assert np.allclose(t.unitcell_lengths, t_ref.unitcell_lengths)

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_force_overwrite(write_traj, monkeypatch):
    if write_traj.fext == "dtr":
        pytest.xfail("This is broken with dtr")

    def test_base(write_traj):
        t_ref = md.load(data_path("native2.pdb"), no_boxchk=True)
        open(write_traj.fn, "w").close()
        t_ref.save(write_traj.fn, force_overwrite=True)

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_force_noverwrite(write_traj, monkeypatch):
    def test_base(write_traj):
        t_ref = md.load(data_path("native2.pdb"), no_boxchk=True)
        open(write_traj.fn, "w").close()
        with pytest.raises(IOError):
            t_ref.save(write_traj.fn, force_overwrite=False)

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_length():
    files = ["frame0.nc", "frame0.xtc", "frame0.trr", "frame0.dcd", "2EQQ.pdb"]

    for file in files:
        opened = md.open(data_path(file))

        if "." + file.rsplit(".", 1)[-1] in biotraj.core.trajectory._TOPOLOGY_EXTS:
            top = file
        else:
            top = "native.pdb"

        loaded = md.load(data_path(file), top=data_path(top))
        assert len(opened) == len(loaded)


def test_unitcell(write_traj, monkeypatch):
    # make sure that bogus unitcell vectors are not saved
    # NOTE: Important if/once these fileformats are readded
    # if write_traj.fext in ["rst7", "ncrst", "lammpstrj", "dtr"]:
    #    pytest.xfail(f"{write_traj.fext} seems to need unit vectors")

    def test_base(write_traj):
        top = md.load(data_path("native.pdb")).restrict_atoms(range(5)).topology
        t = md.Trajectory(xyz=np.random.randn(100, 5, 3), topology=top)
        t.save(write_traj.fn)
        assert md.load(write_traj.fn, top=top).unitcell_vectors is None

    if write_traj.fext in ("nc"):
        # Running with scipy
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "netCDF4", None)
            test_base(write_traj)

    test_base(write_traj)


def test_chunk0_iterload():
    filename = "frame0.pdb"

    trj0 = md.load(data_path(filename))

    for trj in md.iterload(data_path(filename), chunk=0):
        pass

    assert np.allclose(trj0.n_frames, trj.n_frames)


def test_hashing():
    frames = [
        frame
        for frame in md.iterload(
            data_path("frame0.xtc"),
            chunk=1,
            top=data_path("native.pdb"),
        )
    ]
    hashes = [hash(frame) for frame in frames]
    # check all frames have a unique hash value
    assert len(hashes) == len(set(hashes))

    # change topology and ensure hash changes too
    top = frames[0].topology
    top.add_bond(top.atom(0), top.atom(1))

    last_frame_hash = hash(frames[0])
    assert last_frame_hash != hashes[-1]

    # test that trajectories without unitcell data can be hashed
    t1 = md.load(data_path("1bpi.pdb"))
    t2 = md.load(data_path("1bpi.pdb"))
    assert hash(t1) == hash(t2)


def test_smooth():
    from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

    pad = 5
    order = 3
    b, a = butter(order, 2.0 / pad)
    zi = lfilter_zi(b, a)

    signal = np.sin(np.arange(100))
    padded = np.r_[signal[pad - 1 : 0 : -1], signal, signal[-1:-pad:-1]]

    z, _ = lfilter(b, a, padded, zi=zi * padded[0])
    z2, _ = lfilter(b, a, z, zi=zi * z[0])

    output = filtfilt(b, a, padded)
    test = np.loadtxt(data_path("smooth.txt"))

    assert np.allclose(output, test)


@pytest.mark.skip(reason="Broken, maybe only on Python 3.11")
def test_image_molecules():
    # Load trajectory with periodic box
    t = md.load(
        data_path("alanine-dipeptide-explicit.dcd"),
        top=data_path("alanine-dipeptide-explicit.pdb"),
    )
    # Image to new trajectory
    t_new = t.image_molecules(inplace=False)
    # Test that t_new and t are not the same object (issue #1769)
    assert t_new.xyz is not t.xyz
    # Image inplace without making molecules whole
    t.image_molecules(inplace=True, make_whole=False)
    # Image inplace with making molecules whole
    t.image_molecules(inplace=True, make_whole=True)
    # Test coordinates in t are not corrupted to NaNs (issue #1813)
    assert np.any(np.isnan(t.xyz)) is False
    # Image with specified anchor molecules
    molecules = t.topology.find_molecules()
    anchor_molecules = molecules[0:3]
    t.image_molecules(inplace=True, anchor_molecules=anchor_molecules)


def test_load_pdb_no_standard_names():
    # Minimal test. Standard_names=False will force load_pdb.py
    # to NOT replace any non-standard atom or residue names in the topology
    md.load(data_path("native2.pdb"), standard_names=False, no_boxchk=True)
    md.load_pdb(data_path("native2.pdb"), standard_names=False, no_boxchk=True)


def test_load_with_atom_indices():
    t1 = md.load(data_path("frame0.xtc"), top=data_path("frame0.gro"), atom_indices=[5])
    t2 = md.load(data_path("frame0.xtc"), top=data_path("frame0.gro"))
    t2 = t2.atom_slice([5])
    assert np.allclose(t1.xyz, t2.xyz)
    assert np.allclose(t1.time, t2.time)


def test_load_with_frame():
    t1 = md.load(data_path("frame0.xtc"), top=data_path("frame0.pdb"), frame=3)
    t2 = md.load(data_path("frame0.xtc"), top=data_path("frame0.pdb"))
    t2 = t2.slice([3])
    assert np.allclose(t1.xyz, t2.xyz)
    assert np.allclose(t1.time, t2.time)


def test_add_remove_atoms():
    t = md.load(data_path("aaqaa-wat.pdb"))
    top = t.topology
    old_atoms = list(top.atoms)[:]
    # Add an atom 'MW' at the end of each water molecule
    for r in list(top.residues)[::-1]:
        if r.name != "HOH":
            continue
        atoms = list(r.atoms)
        midx = atoms[-1].index + 1
        top.insert_atom("MW", None, r, index=midx)
    mwidx = [a.index for a in list(top.atoms) if a.name == "MW"]
    # Check to see whether the 'MW' atoms have the correct index
    assert mwidx == [183 + 4 * i for i in range(83)]
    # Now delete the atoms again
    for r in list(top.residues)[::-1]:
        if r.name != "HOH":
            continue
        atoms = list(r.atoms)
        top.delete_atom_by_index(atoms[-1].index)
    roundtrip_atoms = list(top.atoms)[:]
    # Ensure the atoms are the same after a round trip of adding / deleting
    assert old_atoms == roundtrip_atoms
