import functools
import glob
import os
from os.path import join, dirname, realpath
import sys
import warnings
from argparse import ArgumentParser

import numpy as np

import biotraj as md
from biotraj.core.trajectory import _parse_topology
from biotraj.utils import in_units_of

def data_dir():
    return join(dirname(realpath(__file__)), "data")