README for pysndlib 
======================= 

This package provides a python wrapper around sndlib, by Bill Schottstaedt (bil@ccrma.stanford.edu)
Sources for sndlib can be found here <https://ccrma.stanford.edu/software/snd/sndlib/>

cython is used to generate bindings and is required

This also requires numpy.

Building sndlib
---------------------

Update recommended build steps

Check out psyndlib from github

grab sndlib from ftp://ccrma-ftp.stanford.edu/pub/Lisp/sndlib.tar.gz

copy sndlib directory to top level of pysndlib

cd sndlib

./configure --with-s7=no --with-gsl=no --without-audio


make


leave libsndlib.a in sndlib directory

this avoids writing over installed versions which might be built with other options and
potential linking issues


Python dependencies
---------------------

I do the following
::

    pip install numpy
    
    pip install Cython
    
    
After grabbing the pysndlib sources i make sure I am in the venv I want and then from top level
of the pysndlib clone:


    python setup.py build_ext -i



:: 
    
    pip install -e .

style

My intention was to be as literal as possible with translation so that it would be easy to port
existing examples. This may mean some aspects may not be pythonic.

if you are already familiar with clm
-------------------------------------

* Underscores replace hyphens

* make_oscil instead of make-oscil

* is_oscil instead of oscil?

* Using 2 instead of -> e.g. hz2radians instead of hz->radians

* Instead of using generic methods for setting and getting use property

    (set! (mus-frequency gen) 100.) => gen.mus_frequency = 100.

    (set! f (mus-frequency gen)) => f = gen.mus_frequency

* there is a simple name space CLM that hold global variables one would use in clm

    e.g. instead of *clm-srate* you would use CLM.srate *clm-table-size* is CLM.table_size

* there are python enums for the sndlib enums. Interp, Window, Spectrum, Polynomial, Header, Sample, Error

    e.g. Window.RECTANGULAR, Header.AIFC

* with-sound is implemented as a context manager and has similar options.
    with Sound(play=True): gen = make_oscil(440.0) for i in range(44100): outa(i, .5 * oscil(gen)) An 'instrument' will just be defined as a function (see examples in clm_ins.py and demos)


* one can use ndarrays from numpy instead of writing to files. the shape for the numpy arrays is channels, length. in other words a mono audio buffer of 44100 samples with be shape (1,44100) this is similar to librosa but opposite of pysndfile

* some clm functions like length or channels have clm prepended to name e.g. clm_length clm_channels.


Many more updates coming. 
