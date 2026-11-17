What's new in 0.8
====================================

Summary of changes in the 0.8.x series. (A detailed internal change
record is kept by the developers; this page is the user-facing
summary.)

Platforms and Python versions
------------------------------------

* **Windows (x86-64) support** via MSVC and the sndlib CMake build;
  pysndlib now provides binary wheels for macOS (universal2), x86-64
  Linux, and x86-64 Windows for Python 3.10 through 3.14.
* Wheels are built, tested, and published from CI; the full test
  suite runs against every wheel before upload.

Settings-file support
------------------------------------

* Settings can live in a platform-appropriate location
  (``%APPDATA%\pysndlib\pysndlib.py`` on Windows,
  ``~/Library/Application Support/pysndlib/`` on macOS, XDG config on
  Linux), or in the legacy ``~/.pysndlib``. ``$PYSNDLIB_RC``
  overrides; ``$PYSNDLIB_RC_STRICT`` makes errors fatal; see
  :doc:`settings`.
* ``pysndlib.clm.load_rc()`` loads (or re-loads) a settings file at
  any time.

Error handling
------------------------------------

* sndlib errors now surface as ``pysndlib.clm.SNDLibError`` raised at
  well-defined points (generator construction, output, file I/O,
  ``Sound.__exit__``) instead of as delayed, hard-to-trace failures.
  ``pysndlib.clm.check_sndlib_errors()`` reports any error recorded
  by the sndlib error handler.
* sndlib ``mus_print`` output is routed to python's stdout.

Playback
------------------------------------

* ``Sound(play=True)`` accepts a ``default.player`` that is either an
  executable name (``'afplay'``, ``'aplay'``) or a command list with a
  ``'{file}'`` placeholder (see :doc:`settings`). Windows has a
  working default player.

Bug fixes
------------------------------------

* ``envelope_interp`` accepts breakpoint lists (as in CLM's
  ``env-interp``), not just envelope generators.
* 64-bit sample counts and lengths no longer truncate on Windows
  (``mus_sound_samples``, ``mus_sound_framples``,
  ``mus_sound_data_location``, ``mus_sound_length``,
  ``clm.get_length``, ``clm.get_framples``).
* ``mus_sound_comment`` no longer crashes on files without a comment.
* Numerous fixes in the shipped instruments and generators (e.g.
  ``MovingFFT`` and ``Noddsin`` were unconstructible;
  ``canter`` now accepts list envelopes).

API changes
------------------------------------

* ``pysndlib.sndlib.mus_sound_reopen_output`` no longer takes a
  ``srate`` parameter (it was never passed to sndlib).
* ``pysndlib.generators.make_nchoosekcos(frequency, ratio, n)`` --
  the previous factory passed its arguments to the constructor in the
  wrong order.
* ``pysndlib.clm.locsig`` returns ``None`` (previously returned an
  unintended ``0.0``).

Development
------------------------------------

* The version number is derived from git tags (setuptools-scm); see
  ``RELEASE.md`` in the repository for the release process.
* A test suite (340+ tests) runs on every supported platform and
  python version in CI.
* A hands-on tutorial series of notebooks is included in the
  repository under ``notebooks/``.
