from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
import numpy

extensions = [
    Extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"], 
        libraries=["sndlib"], 
        extra_compile_args=['-Wno-parentheses-equality'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    Extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], 
        libraries=["sndlib"], include_dirs=[numpy.get_include()], 
        extra_compile_args=['-Wno-parentheses-equality'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

if __name__ == '__main__':

    setup(
         packages=["pysndlib", "pysndlib.clm", "pysndlib.sndlib"],
        ext_modules = cythonize(extensions, compiler_directives={'language_level': '3str'})
    )


