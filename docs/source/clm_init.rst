CLM Initialization
======================

**in pysndlib.clm**

There are a number of default settings when using the library, particularly when using
the Sound context manager. These defaults are stored in a simple-namespace called CLM. 

While many of these can also be set in individual calls to with Sound, they can also be set
globally. For instance, if you want to change the default sample rate to 48kHz put this 
statement in your code.

`CLM.srate = 48000`


.. module:: 
	pysndlib.clm
	:noindex:
		
.. autoattribute:: CLM.file_name

	Output file name

.. autoattribute:: CLM.srate
	
	Output sampling rate

.. autoattribute:: CLM.channels
	
	Channels in output

.. autoattribute:: CLM.sample_type

	Output sample data type 

.. autoattribute:: CLM.header_type

	Output header type

.. autoattribute:: CLM.verbose

	Print out some info

.. autoattribute:: CLM.play
	
	If True, play the sound automatically

.. autoattribute:: CLM.statistics

	If True, print info at end of with-sound (compile time, maxamps)

.. autoattribute:: CLM.reverb

	Reverb instrument

.. autoattribute:: CLM.reverb_channels
	
	Chans in the reverb intermediate file

.. autoattribute:: CLM.reverb_data
	
	Arguments passed to the reverb

.. autoattribute:: CLM.reverb_file_name
	
	Reverb intermediate output file name

.. autoattribute:: CLM.table_size
	
	Default size for wavetables

.. autoattribute:: CLM.buffer_size
	
	Buffer size for file IO

.. autoattribute:: CLM.locsig_type
	
	Locsig panning mode

.. autoattribute:: CLM.clipped
	
	Whether to clip samples if out of range
.. autoattribute:: CLM.player

	Process to use for file playback

.. autoattribute:: CLM.output

	Default output for output gens

.. autoattribute:: CLM.delete_reverb

	If True, delete reverb file


Indices and tables   
==================    

* :ref:`genindex` 
* :ref:`modindex`
* :ref:`search`


