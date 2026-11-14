Installing pysndlib
=========================

If you use pip, you can install pysndlib with:

:: 

    pip install pysndlib
    
    
This currently supports macOs (Intel and Arm), x86-64 Linux, and Windows (x86-64) and python versions 3.10 through 3.14

Settings
=========================

pysndlib settings can be placed in a settings file (an ordinary python script, the analog of clm's ~/.clm-init.lisp) -- typically a sequence of assignments to the ``default`` namespace, e.g. ``default.srate = 48000``. The file is loaded at import time. The first existing file of the following is used:

* ``$PYSNDLIB_RC`` if set (path to any file)
* ``%APPDATA%/pysndlib/pysndlib.py`` on Windows, ``~/Library/Application Support/pysndlib/pysndlib.py`` on macOS, ``$XDG_CONFIG_HOME/pysndlib/pysndlib.py`` (default: ``~/.config/pysndlib/pysndlib.py``) on Linux
* ``~/.pysndlib`` (legacy location, kept for back-compat)

Errors in the settings file are reported as warnings rather than breaking the import; set ``$PYSNDLIB_RC_STRICT=1`` to make them fatal. The file can also be re-loaded at any time with ``pysndlib.clm.load_rc(path)``.
 
I would suggest using a virtual environment. `Here <https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#howto>`_ is a good guide.




