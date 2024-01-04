CLM Initialization
======================

**in pysndlib.clm**

There are a number of default settings when using the library, particularly when using
the Sound context manager. These defaults are stored in a simple-namespace called default. 

While many of these can also be set in individual calls to with Sound, they can also be set
globally. For instance, if you want to change the default sample rate to 48kHz put this 
statement in your code.

Examples:
::

    clm.default.srate = 48000

    clm.default.sample_type = Sample.BDOUBLE

    clm.default.channels = 2



.. module:: 
	pysndlib.clm
	:noindex:
		
.. autoattribute:: default.file_name

	Output file name

.. autoattribute:: default.srate
	
	Output sampling rate

.. autoattribute:: default.channels
	
	Channels in output

.. autoattribute:: default.sample_type

	Output sample data type 

.. autoattribute:: default.header_type

	Output header type

.. autoattribute:: default.verbose

	Print out some info

.. autoattribute:: default.play
	
	If True, play the sound automatically

.. autoattribute:: default.statistics

	If True, print info at end of with-sound (compile time, maxamps)

.. autoattribute:: default.reverb

	Reverb instrument

.. autoattribute:: default.reverb_channels
	
	Chans in the reverb intermediate file

.. autoattribute:: default.reverb_data
	
	Arguments passed to the reverb

.. autoattribute:: default.reverb_file_name
	
	Reverb intermediate output file name

.. autoattribute:: default.table_size
	
	Default size for wavetables

.. autoattribute:: default.buffer_size
	
	Buffer size for file IO

.. autoattribute:: default.locsig_type
	
	Locsig panning mode

.. autoattribute:: default.clipped
	
	Whether to clip samples if out of range
.. autoattribute:: default.player

	Process to use for file playback

.. autoattribute:: default.output

	Default output for output gens

.. autoattribute:: default.delete_reverb

	If True, delete reverb file


