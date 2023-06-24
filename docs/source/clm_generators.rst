CLM Generators
================

oscil
-------

.. module:: 
	pysndlib.clm
	:noindex:

.. autofunction:: make_oscil
.. autofunction:: oscil
.. autofunction:: is_oscil

.. table:: oscil properties
	:align: left
	:width: 60%

	===============  ==================================			
	===============  ==================================
	mus_frequency    frequency in Hz
	mus_phase        phase in radians
	mus_length	     1 (no setable)
	mus_increment    frequency in radians per sample
	===============  ==================================

Example usage

:: 
	
	with Sound(play=True):
		gen = make_oscil(440.0)
		for i in range(44100):
			outa(i,  oscil(gen))


.. seealso::
	`sndlib all-pass <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc>`_



oscil_bank
-----------

.. autofunction:: make_oscil_bank
.. autofunction:: oscil_bank
.. autofunction:: is_oscil_bank


Example usage

:: 
	
	with Sound(play=True, statistics=False):
		phases98 = [0.000000, -0.183194, 0.674802, 1.163820, -0.147489, 1.666302, 0.367236, 0.494059, 0.191339,
					0.714980, 1.719816, 0.382307, 1.017937, 0.548019, 0.342322, 1.541035, 0.966484, 0.936993,
					-0.115147, 1.638513, 1.644277, 0.036575, 1.852586, 1.211701, 1.300475, 1.231282, 0.026079,
					0.393108, 1.208123, 1.645585, -0.152499, 0.274978, 1.281084, 1.674451, 1.147440,0.906901,
					1.137155, 1.467770, 0.851985, 0.437992, 0.762219, -0.417594, 1.884062, 1.725160, -0.230688,
					0.764342, 0.565472, 0.612443, 0.222826, -0.016453, 1.527577, -0.045196, 0.585089, 0.031829,
					0.486579, 0.557276, -0.040985, 1.257633, 1.345950, 0.061737, 0.281650, -0.231535, 0.620583,
					0.504202, 0.817304, -0.010580, 0.584809, 1.234045, 0.840674, 1.222939, 0.685333, 1.651765,
					0.299738, 1.890117, 0.740013, 0.044764, 1.547307, 0.169892, 1.452239, 0.352220, 0.122254,
					1.524772, 1.183705, 0.507801, 1.419950, 0.851259, 0.008092, 1.483245, 0.608598, 0.212267,
					0.545906, 0.255277, 1.784889, 0. 270552, 1.164997, -0.083981, 0.200818, 1.204088]
				
		freq = 10.0
		dur = 5.0
		n = 98
	
		samps = math.floor(dur * 44100)
		onedivn = 1.0 / n
		freqs = np.zeros(n)
		phases = np.zeros(n)
		phases.fill(math.pi*.5)
		for i in range(n):
			off = (math.pi * (.5 - phases98[i])) / dur / 44100
			h = hz2radians(freq*(i+1))
			freqs[i] = h + off
	
		ob = make_oscil_bank(freqs, phases)
		for i in range(1000): #get rid of the distracting initial click
			nada = oscil_bank(ob) # added in this assignment because otherwise was printing output
		for k in range(samps):
			outa(k, onedivn * oscil_bank(ob))


.. seealso::
	`sndlib all-pass <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc>`_



env
-------

.. module:: 
	pysndlib.clm
	:noindex:

.. autofunction:: make_env
.. autofunction:: env
.. autofunction:: is_env

.. table:: env properties
	:align: left
	:width: 60%

	===============  ==================================			
	===============  ==================================
	mus_location     number of calls so far on this env
	mus_increment    base
	mus_data         original breakpoint list
	mus_scaler       scaler
	mus_offset       offset
	mus_length       duration in samples
	mus_channels     current position in the break-point list
	===============  ==================================

Example usage

:: 
	
	with Sound(play=True):
		gen = make_oscil(440.0)
		ampf = make_env([0., 0., 0.01, 1.0, 0.25, 0.1, 1, 0], scaler=.5, length=44100)
		for i in range(44100):
			outa(i,  env(ampf)*oscil(gen))


.. seealso::
	`sndlib env https://ccrma.stanford.edu/software/snd/snd/sndclm.html#envdoc>`_




all_pass 
---------

.. autofunction:: make_all_pass
.. autofunction:: all_pass
.. autofunction:: is_all_pass

.. table:: all_pass properties
	:align: left
	:width: 60%

	===============  ==================================			
	===============  ==================================
	mus_length       length of delay
	mus_order        same as mus-length
	mus_data	     delay line itself (no setable)
	mus_feedback     feedback scaler
	mus_feedforward  feedforward scaler
	mus_interp-type  interpolation choice (no setable)
	===============  ==================================

Example usage

:: 
	
	with Sound(play=True):
		alp = make_all_pass(-0.4, 0.4, seconds2samples(0.4))
		osc = make_oscil(440.0)
		ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
		for i in range(88200):
			outa(i, 0.5 * (all_pass(alp, env(ampf) * oscil(osc))))


.. seealso::
	`sndlib all-pass <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#all-passdoc>`_


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


