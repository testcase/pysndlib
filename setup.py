from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

import os
os.environ['LDFLAGS'] = '-framework CoreAudio -framework CoreFoundation -framework CoreMIDI'

extensions = [
    Extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"], 
        extra_objects=["/usr/local/lib/libsndlib.a"], 
        extra_compile_args=['-Wno-parentheses-equality'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    Extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], 
        extra_objects=["/usr/local/lib/libsndlib.a"], include_dirs=[numpy.get_include()], 
        extra_compile_args=['-Wno-parentheses-equality'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

if __name__ == '__main__':

    setup(

        ext_modules = cythonize(extensions, compiler_directives={'language_level': '3str', 'embedsignature' : False})
    )



# extensions = [
#     Extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"], 
#         libraries=["sndlib"], 
#         extra_compile_args=['-Wno-parentheses-equality'],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
#     Extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], 
#         libraries=["sndlib"], include_dirs=[numpy.get_include()], 
#         extra_compile_args=['-Wno-parentheses-equality'],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
# ]
# 
# if __name__ == '__main__':
# 
#     setup(
# 
#         ext_modules = cythonize(extensions, compiler_directives={'language_level': '3str', 'embedsignature' : False})
#     )


