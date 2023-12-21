from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False
import numpy
import sys
import os
# 
# linking to static lib
# 
if sys.platform.startswith("darwin"):
    os.environ['LDFLAGS'] = '-framework CoreAudio -framework CoreFoundation -framework CoreMIDI'
    os.environ['CFLAGS'] = '-arch x86_64 -arch arm64'
    

    extensions = [
        Extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'],
            extra_compile_args=['-Wno-parentheses-equality' ],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.env", ["src/pysndlib/env.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.clm_ins", ["src/pysndlib/clm_ins.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.v", ["src/pysndlib/v.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.jcrev", ["src/pysndlib/jcrev.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.dsp", ["src/pysndlib/dsp.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.birds", ["src/pysndlib/birds.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.generators", ["src/pysndlib/generators.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.jcvoi", ["src/pysndlib/jcvoi.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    
    ]


if sys.platform.startswith("linux"):
    os.environ['LDFLAGS'] = '-lm -ldl'

    extensions = [
        Extension("pysndlib.sndlib", ["src/pysndlib/sndlib.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'],
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.clm", ["src/pysndlib/clm.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.env", ["src/pysndlib/env.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.clm_ins", ["src/pysndlib/clm_ins.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.v", ["src/pysndlib/v.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.jcrev", ["src/pysndlib/jcrev.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.dsp", ["src/pysndlib/dsp.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.birds", ["src/pysndlib/birds.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.generators", ["src/pysndlib/generators.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("pysndlib.jcvoi", ["src/pysndlib/jcvoi.pyx"], 
            extra_objects=["./sndlib/libsndlib.a"], 
            include_dirs=[numpy.get_include(), './sndlib'], 
            extra_compile_args=['-Wno-parentheses-equality', '-Wno-unreachable-code-fallthrough'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ]


if __name__ == '__main__':

    setup(

        ext_modules = cythonize(extensions, compiler_directives={'language_level': '3str', 'embedsignature' : False})
    )


