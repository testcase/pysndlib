Settings File
====================================

pysndlib can be customized with a settings file -- an ordinary python
file that is executed when ``pysndlib.clm`` is imported. It is the
analog of clm's ``~/.clm-init.lisp``: typically a sequence of
assignments to the ``default`` namespace, e.g.::

    default.srate = 48000

The file is executed in the ``pysndlib.clm`` module namespace, so bare
``default.x = ...`` statements (as well as any ordinary python) work.
It can also be re-loaded at any time with
:func:`pysndlib.clm.load_rc`::

    >>> import pysndlib.clm as clm
    >>> clm.load_rc()          # search the default locations again
    True
    >>> clm.load_rc("my_settings.py")   # load a specific file
    True

``load_rc`` returns ``True`` if a file was found and executed.

Where the file can be located
------------------------------------

The first existing file in the following order is used:

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Priority
     - Location
     - Notes
   * - 1
     - ``$PYSNDLIB_RC``
     - explicit override: a path to any file
   * - 2
     - **Windows:** ``%APPDATA%\pysndlib\pysndlib.py``
     - Roaming profile (fallback: ``~/AppData/Roaming``)
   * - 2
     - **macOS:** ``~/Library/Application Support/pysndlib/pysndlib.py``
     - 
   * - 2
     - **Linux:** ``$XDG_CONFIG_HOME/pysndlib/pysndlib.py``
     - default: ``~/.config/pysndlib/pysndlib.py``
   * - 3
     - ``~/.pysndlib``
     - legacy location; works on every platform

Errors in the settings file are reported as warnings rather than
breaking the import; a broken file can never make importing pysndlib
fail. Set ``$PYSNDLIB_RC_STRICT=1`` to make errors fatal instead.
Tracebacks name the settings file, so errors are easy to find.

Options
------------------------------------

The settings are attributes of the ``pysndlib.clm.default``
namespace. All of the following can be set in the settings file:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Option
     - Default
     - Meaning
   * - ``default.srate``
     - ``44100``
     - default output sampling rate
   * - ``default.channels``
     - ``1``
     - default number of output channels
   * - ``default.sample_type``
     - ``sndlib.Sample.LFLOAT``
     - default output sample type
   * - ``default.header_type``
     - ``sndlib.Header.RIFF``
     - default output header type
   * - ``default.file_name``
     - ``'test.wav'``
     - default output file name
   * - ``default.player``
     - platform-dependent (see below)
     - player used by ``Sound(play=True)``
   * - ``default.table_size``
     - ``512``
     - default wavetable size
   * - ``default.buffer_size``
     - ``65536``
     - file I/O buffer size
   * - ``default.locsig_type``
     - ``clm.Interp.LINEAR``
     - locsig interpolation type
   * - ``default.clipped``
     - ``False``
     - whether output is clipped
   * - ``default.statistics``
     - ``False``
     - default for ``Sound(statistics=True)``
   * - ``default.play``
     - ``False``
     - default for ``Sound(play=True)``
   * - ``default.verbose``
     - ``False``
     - verbose output
   * - ``default.reverb_channels``
     - ``1``
     - default reverb channel count
   * - ``default.reverb_data``
     - ``None``
     - default reverb instrument parameters
   * - ``default.reverb_file_name``
     - ``'test.rev'``
     - default intermediate reverb file
   * - ``default.delete_reverb``
     - ``False``
     - remove the intermediate reverb file when done

The ``player`` option
------------------------------------

``default.player`` is invoked as a subprocess to play the rendered
sound when ``Sound(play=True)`` (see :doc:`with_sound`). It can be:

* a **string**: the name of an executable that receives the sound file
  as its only argument -- ``'afplay'`` on macOS, ``'aplay'`` on Linux,
  or any player you have installed (e.g. ``'ffplay'``).
* a **list**: a command where a ``'{file}'`` element is replaced by
  the (single-quote escaped) sound file name; if no element uses the
  placeholder, the file name is appended as the last argument. This
  form is needed for players with flags, e.g.::

      default.player = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']

The default on Windows is a list-form command that hosts the Windows
Media Player COM engine through PowerShell -- it plays float WAVs,
runs headless, and waits for playback to finish. There is no default
player equivalent of ``afplay``/``aplay`` on Windows stock installs,
so the PowerShell command is used; ``Sound(play=True)`` raises a
``RuntimeError`` if ``default.player`` is set to ``None``.

Examples
------------------------------------

A settings file for 48 kHz output with statistics on::

    default.srate = 48000
    default.statistics = True

Cross-platform settings (e.g. a different player per OS)::

    import sys
    if sys.platform == 'darwin':
        default.player = 'afplay'
    elif sys.platform.startswith('linux'):
        default.player = 'aplay'
    else:
        default.player = ['ffplay', '-nodisp', '-autoexit']

Since the file is ordinary python, imports, conditionals, and any
other logic are available.

Environment variables
------------------------------------

``PYSNDLIB_RC``
    Path to the settings file to use, overriding all locations above.

``PYSNDLIB_RC_STRICT``
    Set to ``1`` to make errors in the settings file raise at import
    time instead of warning.
