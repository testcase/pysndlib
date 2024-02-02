CLM Utility Functions
======================

.. module:: 
	pysndlib.clm
	:noindex:

General		
--------------
.. autofunction:: get_length
.. autofunction:: random
.. autofunction:: get_channels

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
.. autofunction:: make_fft_window
.. autofunction:: fft
.. autofunction:: rectangular2polar
.. autofunction:: rectangular2magnitudes
.. autofunction:: polar2rectangular
.. autofunction:: spectrum
.. autofunction:: autocorrelate
.. autofunction:: correlate
.. autofunction:: mus_fft
.. autofunction:: mus_rectangular2polar
.. autofunction:: mus_rectangular2magnitudes
.. autofunction:: mus_polar2rectangular
.. autofunction:: mus_convolution
.. autofunction:: mus_spectrum
.. autofunction:: mus_autocorrelate
.. autofunction:: mus_correlate
.. autofunction:: mus_cepstrum

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




