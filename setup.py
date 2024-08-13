import sys

from setuptools import Extension, find_packages, setup

from basesetup import (
    CompilerDetection,
    build_ext,
    parse_setuppy_commands,
    write_version_py,
)

sys.path.insert(0, ".")

try:
    # add an optional --disable-openmp to disable OpenMP support
    sys.argv.remove("--disable-openmp")
    disable_openmp = True
except ValueError:
    disable_openmp = False


##########################
VERSION = "0.1"
ISRELEASED = False
__version__ = VERSION
##########################

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
        extra_compile_args=compiler_args,
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
        "biotraj.dcd",
        sources=[
            "src/biotraj/src/dcdplugin.c",
            "src/biotraj/dcd.pyx",
        ],
        include_dirs=[
            "src/biotraj/include/",
            "src/biotraj/",
        ],
        extra_compile_args=compiler_args,
    )

    return [xtc, trr, dcd]


write_version_py(VERSION, ISRELEASED, "src/biotraj/version.py")

metadata = dict(
    version=__version__,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
            # if _c.__version__ < "0.29":
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
