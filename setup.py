"""MDTraj: A modern, open library for the analysis of molecular dynamics trajectories

MDTraj is a python library that allows users to manipulate molecular dynamics
(MD) trajectories and perform a variety of analyses, including fast RMSD,
solvent accessible surface area, hydrogen bonding, etc. A highlight of MDTraj
is the wide variety of molecular dynamics trajectory file formats which are
supported, including RCSB pdb, GROMACS xtc, and trr, CHARMM / NAMD dcd, AMBER
AMBER NetCDF, AMBER mdcrd, TINKER arc and MDTraj HDF5.
"""

import sys

from setuptools import Extension, find_packages, setup

from basesetup import (
    CompilerDetection,
    StaticLibrary,
    build_ext,
    parse_setuppy_commands,
    write_version_py,
)

DOCLINES = __doc__.split("\n")

sys.path.insert(0, ".")

try:
    # add an optional --disable-openmp to disable OpenMP support
    sys.argv.remove("--disable-openmp")
    disable_openmp = True
except ValueError:
    disable_openmp = False


##########################
VERSION = "1.9.10.dev0"
ISRELEASED = False
__version__ = VERSION
##########################


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# Global info about compiler
compiler = CompilerDetection(disable_openmp)
compiler.initialize()

extra_cpp_libraries = []

if sys.platform == "win32":
    extra_cpp_libraries.append("Ws2_32")
    # For determining if a path is relative (for dtr)
    extra_cpp_libraries.append("Shlwapi")


################################################################################
# Declaration of the compiled extension modules (cython + c)
################################################################################


def format_extensions():
    compiler_args = compiler.compiler_args_warn

    xtc = Extension(
        "biotraj.formats.xtc",
        sources=[
            "src/biotraj/formats/src/xdrfile.c",
            "src/biotraj/formats/src/xdr_seek.c",
            "src/biotraj/formats/src/xdrfile_xtc.c",
            "src/biotraj/formats/xtc.pyx",
        ],
        include_dirs=[
            "src/biotraj/formats/include/",
            "src/biotraj/formats/",
        ],
        extra_compile_args=compiler_args,
    )

    trr = Extension(
        "biotraj.formats.trr",
        sources=[
            "src/biotraj/formats/src/xdrfile.c",
            "src/biotraj/formats/src/xdr_seek.c",
            "src/biotraj/formats/src/xdrfile_trr.c",
            "src/biotraj/formats/trr.pyx",
        ],
        include_dirs=[
            "src/biotraj/formats/include/",
            "src/biotraj/formats/",
        ],
        extra_compile_args=compiler_args,
    )

    zlib_include_dirs = []
    zlib_library_dirs = []
    if sys.platform == "win32":
        # Conda puts the zlib headers in ./Library/... on windows
        # If you're not using conda, good luck!
        zlib_include_dirs += [f"{sys.prefix}/Library/include"]
        zlib_library_dirs += [f"{sys.prefix}/Library/lib"]
    else:
        # On linux (and mac(?)) these paths should work for a standard
        # install of python+zlib or a conda install of python+zlib
        zlib_include_dirs += [f"{sys.prefix}/include"]
        zlib_library_dirs += [f"{sys.prefix}/lib"]

    dcd = Extension(
        "biotraj.formats.dcd",
        sources=[
            "src/biotraj/formats/src/dcdplugin.c",
            "src/biotraj/formats/dcd.pyx",
        ],
        include_dirs=[
            "src/biotraj/formats/include/",
            "src/biotraj/formats/",
        ],
        extra_compile_args=compiler_args,
    )

#    dtr = Extension(
#        "biotraj.formats.dtr",
#        sources=[
#            "src/biotraj/formats/dtr/src/dtrplugin.cxx",
#            "src/biotraj/formats/dtr/dtr.pyx",
#        ],
#        include_dirs=[
#            "src/biotraj/formats/dtr/include/",
#            "src/biotraj/formats/dtr/",
#        ],
#        define_macros=[("DESRES_READ_TIMESTEP2", 1)],
#        language="c++",
#        extra_compile_args=compiler_args,
#        libraries=extra_cpp_libraries,
#    )

    return [xtc, trr, dcd]#, dtr]


write_version_py(VERSION, ISRELEASED, "src/biotraj/version.py")

metadata = dict(
    name="mdtraj",
    author="Robert McGibbon",
    author_email="rmcgibbo@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=__version__,
    license="LGPLv2.1+",
    url="http://biotraj.org",
    download_url="https://github.com/rmcgibbo/src/biotraj/releases/latest",
    platforms=["Linux", "Mac OS-X", "Unix", "Windows"],
    python_requires='>=3.9',
    classifiers=CLASSIFIERS.splitlines(),
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy>=2.0",
        "scipy",
        "pyparsing",
        "packaging",
    ],
    package_data={"biotraj.formats.pdb": ["data/*"]},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "mdconvert = biotraj.scripts.mdconvert:entry_point",
            "mdinspect = biotraj.scripts.mdinspect:entry_point",
        ],
    },
)


if __name__ == "__main__":
    # Don't use numpy if we are just - non-build actions are required to succeed
    # without NumPy for example when pip is used to install Scipy when
    # NumPy is not yet present in the system.
    run_build = parse_setuppy_commands()
    if run_build:
        extensions = format_extensions()

        # most extensions use numpy, add headers for it.
        try:
            import Cython as _c
            from Cython.Build import cythonize
            #if _c.__version__ < "0.29":
            #    raise ImportError("Too old")
        except ImportError as e:
            print(
                "mdtrajs setup depends on Cython (>=3.0). Install it prior invoking setup.py",
            )
            print(e)
            sys.exit(1)
        try:
            import numpy as np
        except ImportError:
            print("biotraj setup depends on NumPy. Install it prior invoking setup.py")
            sys.exit(1)

        for e in extensions:
            e.include_dirs.append(np.get_include())
        metadata["ext_modules"] = cythonize(
            extensions,
            language_level=sys.version_info[0],
            force=True,
        )

    setup(**metadata)
