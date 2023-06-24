# pysndlib
****This is updated to use cython but many things are not working as of yet.****

This package provides a python wrapper around sndlib, by Bill Schottstaedt (bil@ccrma.stanford.edu)
Sources for sndlib can be found [here](https://ccrma.stanford.edu/software/snd/sndlib/)

cython is used to generate bindings

This also requires numpy.

*IMPORTANT* 
At least on macos, this needs to be linked against sndlib built in specific way

sndlib is normally build as an .so as a loadable bundle with the -bundle flag to ld. Cython needs to link
against a dylib built with -dynamiclib flag. Technically I believe it would still work with .so extension but 
I have changed this to make sure I can keep different versions straight.



###run###

`./configure --without-s7`

###edit the makefile###

LDSO_FLAGS line should look like this:


`LDSO_FLAGS = -dynamic -dynamiclib -undefined suppress -flat_namespace`


Change SO_NAME line to this:

`SO_NAME = libsndlib.dylib`

changed install and uninstall targets to following:

`install: sndlib
	$(mkinstalldirs) $(bindir)
	$(mkinstalldirs) $(libdir)
	$(mkinstalldirs) $(includedir)
	$(SO_INSTALL) libsndlib.dylib $(libdir)/libsndlib.dylib
	$(A_INSTALL) libsndlib.a $(libdir)/libsndlib.a
	$(INSTALL) sndlib.h $(includedir)/sndlib.h
	$(INSTALL) sndlib-config $(bindir)/sndlib-config
	$(INSTALL) sndlib.pc $(pkgconfigdir)/sndlib.pc`
	
`uninstall:
	rm -f $(libdir)/libsndlib.dylib
	rm -f $(libdir)/libsndlib.a`




`pip install -e .
`

## style

My intention was to be as literal as possible with translation so that it would be easy to port existing examples. This may
mean some aspects may not be pythonic. 


## if you are already familiar with clm 

- Underscores replace hyphens 

- `make_oscil` instead of `make-oscil`

- `is_oscil` instead of `oscil?`

- Using 2 instead of -> e.g. `hz2radians` instead of `hz->radians`

- Instead of using generic methods for setting and getting use property
	
	`(set! (mus-frequency gen) 100.)`  => `gen.mus_frequency = 100.`
	
	`(set! f (mus-frequency gen))` => `f = gen.mus_frequency`
	
- there is a simple name space CLM that hold global variables one would use in clm
	
	e.g. instead of \*clm-srate\* you would use CLM.srate
	\*clm-table-size\* is CLM.table_size
	
- there are python enums for the sndlib enums. Interp, Window, Spectrum, Polynomial, Header, Sample, Error
	
	e.g. Window.RECTANGULAR, Header.AIFC
	
- with-sound is implemented as a context manager and has similar options.

		with Sound(play=True): 
			gen = make_oscil(440.0)
			for i in range(44100):
				outa(i, .5 * oscil(gen))
	    

An 'instrument' will just be defined as a function (see examples in clm_ins.py and demos)

- Errors need a lot more work so ....

- one can use ndarrays from numpy instead of writing to files. the shape 
for the numpy arrays is channels, length. in other words a mono audio buffer of 
44100 samples with be shape (1,44100) this is similar to librosa but opposite of 
pysndfile

- some clm functions like `length` or `channels` have clm prepended to name e.g. clm_length
 clm_channels. 


 - clm_filter to run filter unclear what to do with this name clash.

## experimental


make_generator can be used to create generators. Python lacks macros so this works a little different

make_generator returns two functions, the function to make the generator and the function to test if generator 
is a type. see generators.py for examples

There is an experimental integration with [musx](https://github.com/musx-admin/musx) by Rick Taube which is being included
in this package at the moment. It uses a decorator to translate a simple function into something useful for musx. The ability
to write a 'score' to a file is not completed yet. Look in demos for examples


	    


