import sys

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

# sndlib static library and headers are expected in ./sndlib
# (see docs/source/building.rst)
LIBSNDBLIB = './sndlib/libsndlib.a'
INCLUDE_DIRS = [numpy.get_include(), './sndlib']
MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# ------------------------------------------------------------------
# platform-specific compile/link flags.
# The extensions are defined once below; per-platform flags are
# attached through per-Extension arguments (distutils/MSVC-safe)
# instead of mutating the CFLAGS/LDFLAGS environment variables.
# ------------------------------------------------------------------

_is_darwin = sys.platform.startswith("darwin")
_is_linux = sys.platform.startswith("linux")

compile_args = []
link_args = []
math_libs = []

if _is_darwin:
    # universal2 (x86_64 + arm64) builds; CoreAudio frameworks are
    # only needed if sndlib is built with audio support.
    compile_args += ['-arch', 'x86_64', '-arch', 'arm64']
    link_args += ['-arch', 'x86_64', '-arch', 'arm64',
                  '-framework', 'CoreAudio',
                  '-framework', 'CoreFoundation',
                  '-framework', 'CoreMIDI']
    math_libs = ['m']
elif _is_linux:
    link_args += ['-lm', '-ldl']

if _is_darwin or _is_linux:
    # silence warnings in the Cython-generated C code for clang/gcc
    compile_args += ['-Wno-parentheses-equality',
                     '-Wno-unreachable-code-fallthrough']


def make_extension(name, sources, **kwargs):
    return Extension(
        name,
        sources,
        extra_objects=[LIBSNDBLIB],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        **kwargs)


extensions = [
    make_extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"]),
    make_extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], libraries=math_libs),
    make_extension("pysndlib.clm_ins", ["src/pysndlib/clm_ins.pyx"], libraries=math_libs),
    make_extension("pysndlib.env", ["src/pysndlib/env.pyx"]),
    make_extension("pysndlib.dsp", ["src/pysndlib/dsp.pyx"]),
    make_extension("pysndlib.generators", ["src/pysndlib/generators.pyx"]),
    make_extension("pysndlib.analog_filter", ["src/pysndlib/analog_filter.pyx"]),
    # NOTE: the "*" module name is a cythonize glob pattern: it expands to one
    # extension per instrument file (pysndlib.instruments.<name>, ...).
    make_extension("*", ["src/pysndlib/instruments/*.py"]),
]


if __name__ == '__main__':

    setup(
        zip_safe=False,
        name = 'pysndlib',
        package_data={'pysndlib': ["*.pyx", "*.py", "*.snd", "*.aiff", "*.wav", "*.pxd", "*.h"]},
        ext_modules = cythonize(extensions, compiler_directives={'language_level': '3', 'embedsignature' : False}, annotate=False)
    )
