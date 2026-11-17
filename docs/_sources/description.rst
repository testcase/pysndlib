Description
================

This package provides a python wrapper around sndlib, by Bill Schottstaedt (bil@ccrma.stanford.edu)
Sources for sndlib can be found `here <https://ccrma.stanford.edu/software/snd/sndlib/>`_

Some definitions:

**sndlib** is library by Bill Schottstaedt written in C. sndlib also includes builtin support for some extension languages 
including `s7 <https://ccrma.stanford.edu/software/snd/snd/s7.html>`_, a Scheme interpreter,  which is also written by Bill Schottstaedt. sndlib can be thought of as having two
main parts, the first is access to audio I/O, including both file and hardware ports. sndlib is able to read a large number of files types and formats. look at the enums
in :doc:`sndlib_enums` to get a sense of file I/O options. the second is called clm which was originally an acronym for Common Lisp Music, is a sound synthesis package 
in the Music V family. It is still `available <https://ccrma.stanford.edu/software/clm/>`_  for Common Lisp. This part of sndlib contains many sound generators and 
are listed in :doc:`clm_generators`. Bill also has a sound editor called `snd <https://ccrma.stanford.edu/software/snd/>`_ which has s7 embedded in it. Throughout these 
various projects are many fascinating examples and techniques and I hope that this project makes these accessible to people who are using Python.

Another project to look at is Rick Taube's `musx <https://musx-admin.github.io/musx/index.html>`_ which is a "is a package for composing and processing symbolic music information"
which is especially usefule for anyone wanting to use pysndlib in a music context.

It is important to understand from the outset that pysndlib is focused on the synthesis and processing of sound data to/from files or `numpy <https://numpy.org>`_ arrays. One writes functions which process audio data
a sample at a time. sndlib itself is fast and highly optimized but Python places some constraints on speed. This project relies on `cython <https://cython.org>`_ to interface with sndlib as well
as to speed up many operations. You do not need to know cython syntax in order to use pysndlib but more involved work can take advantage of the speed improvements offered by cython. I hope to provide
a guide for doing so in the near future. 

My intention was to be as literal as possible with translation so that it would be easy to port
existing examples. This may mean some aspects of pysndlib may not be pythonic. 





if you are already familiar with clm
-------------------------------------

* Underscores replace hyphens
    make_oscil instead of make-oscil

* is_oscil instead of oscil?

* Using 2 instead of -> e.g. hz2radians instead of hz->radians

* Instead of using generic methods for setting and getting use property

    (set! (mus-frequency gen) 100.) => gen.mus_frequency = 100.

    (set! f (mus-frequency gen)) => f = gen.mus_frequency

* there is a simple name space default that hold global variables one would use in clm

    e.g. instead of *clm-srate* you would use clm.default.srate *clm-table-size* is clm.default.table_size

* there are python enums for the sndlib enums. Interp, Window, Spectrum, Polynomial, Header, Sample, Error

    e.g. Window.RECTANGULAR, Header.AIFC

* with-sound is implemented as a context manager and has similar options.
    with Sound(play=True): gen = make_oscil(440.0) for i in range(44100): outa(i, .5 * oscil(gen)) An 'instrument' will just be defined as a function (see examples in clm_ins.py and demos)


* one can use ndarrays from numpy instead of writing to files. the shape for the numpy arrays is channels, length. in other words a mono audio buffer of 44100 samples with be shape (1,44100) this is similar to librosa but opposite of pysndfile



Many more updates coming. 

