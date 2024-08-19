import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_extensions():
    xtc = Extension(
        "biotraj.xtc",
        sources=[
            "src/biotraj/src/xdrfile.c",
            "src/biotraj/src/xdr_seek.c",
            "src/biotraj/src/xdrfile_xtc.c",
            "src/biotraj/xtc.pyx",
        ],
        include_dirs=[
            "src/biotraj/include/",
            "src/biotraj/",
        ],
    )

    trr = Extension(
        "biotraj.trr",
        sources=[
            "src/biotraj/src/xdrfile.c",
            "src/biotraj/src/xdr_seek.c",
            "src/biotraj/src/xdrfile_trr.c",
            "src/biotraj/trr.pyx",
        ],
        include_dirs=[
            "src/biotraj/include/",
            "src/biotraj/",
        ],
    )

    dcd = Extension(
        "biotraj.dcd",
        sources=[
            "src/biotraj/src/dcdplugin.c",
            "src/biotraj/dcd.pyx",
        ],
        include_dirs=[
            "src/biotraj/include/",
            "src/biotraj/",
        ],
    )

    extensions = [xtc, trr, dcd]
    # Configure NumPy for all of these extensions
    for e in extensions:
        e.include_dirs.append(np.get_include())
        e.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    return extensions


try:
    extensions = cythonize(
        get_extensions(),
        language_level=3,
    )
except ValueError:
    # This is a source distribution and the directory already contains
    # only C/C++ files
    extensions = get_extensions()

setup(
    zip_safe=False,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=extensions,
)
