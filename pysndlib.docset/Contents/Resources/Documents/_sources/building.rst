Building pysndlib
=========================

Building sndlib
---------------------

Check out psyndlib from github

::

    git clone https://github.com/testcase/pysndlib.git
    
    
    cd pysndlib
    

grab sndlib from ftp://ccrma-ftp.stanford.edu/pub/Lisp/sndlib.tar.gz

:: 

    wget ftp://ccrma-ftp.stanford.edu/pub/Lisp/sndlib.tar.gz

copy sndlib directory to top level of pysndlib

:: 
    
    cd sndlib


Configure and make with the following options:

::

    ./configure --with-s7=no --with-gsl=no --without-audio


    make


Leave libsndlib.a in sndlib directory. this avoids writing over installed versions which might be built with other options and
potential linking issues


Python dependencies
---------------------

You must install numpy and cython to build pysndlib

::

    pip install numpy
    
    pip install Cython
    

::

    python setup.py build_ext -i

:: 
    
    pip install -e .

