from os import path

import mdtraj as md

traj = md.load(path.join("tests", "data", "traj.h5"))
first_frame = traj[0]

traj.save(path.join("tests", "data", "traj_prep.xtc"))
first_frame.save(path.join("tests", "data", "traj_prep_top.pdb"))