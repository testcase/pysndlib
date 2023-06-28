CLM Utility Functions
======================

.. module:: 
	pysndlib.clm
	:noindex:


Sampling rate		
--------------
.. autofunction:: get_srate
.. autofunction:: set_srate		

Conversions		
------------
.. autofunction:: radians2hz
.. autofunction:: hz2radians
.. autofunction:: degrees2radians
.. autofunction:: radians2degrees
.. autofunction:: db2linear
.. autofunction:: linear2db
.. autofunction:: seconds2samples
.. autofunction:: samples2seconds

Even/Odd multiples and weighting		
---------------------------------
.. autofunction:: odd_multiple
.. autofunction:: even_multiple
.. autofunction:: odd_weight
.. autofunction:: even_weight




.. autofunction:: ring_modulate
.. autofunction:: amplitude_modulate
.. autofunction:: contrast_enhancement
.. autofunction:: dot_product
.. autofunction:: polynomial
.. autofunction:: array_interp
.. autofunction:: bessi0
.. autofunction:: mus_interpolate

FFT and Convolution		
----------------------

.. autofunction:: fft
.. autofunction:: make_fft_window
.. autofunction:: rectangular2polar
.. autofunction:: rectangular2magnitudes
.. autofunction:: polar2rectangular
.. autofunction:: convolution

.. autofunction:: spectrum

.. autofunction:: autocorrelate
.. autofunction:: correlate
.. autofunction:: cepstrum

Partials	
----------

.. autofunction:: partials2wave
.. autofunction:: phase_partials2wave
.. autofunction:: partials2polynomial
.. autofunction:: normalize_partials

Chebyshev	
----------
.. autofunction:: chebyshev_tu_sum
.. autofunction:: chebyshev_t_sum
.. autofunction:: chebyshev_u_sum



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

