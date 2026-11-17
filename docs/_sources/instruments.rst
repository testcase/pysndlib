CLM Instruments
====================================

pysndlib ships the classic CLM instrument collection. An instrument
is simply a function that is called inside a :class:`pysndlib.clm.Sound`
context and writes samples (see :doc:`with_sound`)::

    import pysndlib.clm as clm
    from pysndlib.instruments import fm_violin

    with clm.Sound('violin.wav', channels=1, srate=44100):
        fm_violin(0.0, 2.0, 440.0, 0.4)

Instruments can be freely mixed in one ``Sound``; each writes into the
current output at its own start time.

Where to find them
------------------------------------

The instruments live in two modules:

* ``pysndlib.instruments`` -- the classic collection (~48
  instruments, ported from clm-ins.scm, generators.scm, and related
  files in the sndlib distribution). The instrument files are
  compiled with cython when pysndlib is built, so they run at native
  speed.
* ``pysndlib.clm_ins`` -- additional translations (``vox``,
  ``drone``, ``canter``).

.. note::

   Import instruments directly, e.g.::

      from pysndlib.instruments import fm_violin

   Do **not** use ``import pysndlib.instruments.fm_violin`` -- the
   package ``__init__`` re-exports each instrument function, which
   shadows the submodule of the same name; the plain import binds the
   function rather than the module.

Reverb instruments
------------------------------------

The reverb instruments (``jc_reverb``, ``nrev``, ``jl_reverb``) are
not called directly: they are handed to ``Sound`` via the ``reverb``
argument, and pysndlib runs them over an intermediate reverb file::

    from pysndlib.instruments import jc_reverb, fm_bell

    with clm.Sound('bells.wav', channels=1, srate=44100,
                   reverb=jc_reverb, reverb_data={'decay_time': 2.0},
                   reverb_channels=1, revfile='bells.rev'):
        fm_bell(0.0, 2.0, 440.0, 0.4)

The instrument table
------------------------------------

Descriptions follow the original sndlib sources (clm-ins.scm,
generators.scm, and the STK-derived instruments by Perry Cook).

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Instrument
     - Description
     - Module
   * - ``attract``
     - attractor-based additive synthesis (clm-ins.scm)
     - ``pysndlib.instruments``
   * - ``bird``
     - a bird song from an envelope-encoded spectrogram (bird.scm;
       module ``pysndlib.instruments.birds``)
     - ``pysndlib.instruments``
   * - ``bouncy``
     - SSB tone with a bouncing (vibrato-rate) modulation
     - ``pysndlib.instruments``
   * - ``brassy``
     - brass-like tone via the ``rxykcos`` generator
     - ``pysndlib.instruments``
   * - ``canter``
     - brass instrument whose brightness follows its amplitude (FM)
     - ``pysndlib.clm_ins``
   * - ``cellon``
     - cello-like tones (STK CelloN: pulse train through a bank of
       formant resonators)
     - ``pysndlib.instruments``
   * - ``clm_expsrc``
     - sampling-rate conversion of a sound file with an envelope
       controlling the conversion ratio
     - ``pysndlib.instruments``
   * - ``drone``
     - slowly evolving waveform drone (wavetable lookup with
       modulated vibrato)
     - ``pysndlib.clm_ins``
   * - ``ercoser``
     - a ser tone based on the raised-cosine (``ercos``) generator
     - ``pysndlib.instruments``
   * - ``exp_snd``
     - granular expansion of a sound file (clm-ins exp-snd)
     - ``pysndlib.instruments``
   * - ``fm_bell``
     - inharmonic FM bell with a decaying index envelope
     - ``pysndlib.instruments``
   * - ``fm_drum``
     - FM drum: rapid index sweep on a low carrier
     - ``pysndlib.instruments``
   * - ``fm_insect``
     - insect wing tone via FM
     - ``pysndlib.instruments``
   * - ``fm_trumpet``
     - two-note brass FM with brightness following amplitude
     - ``pysndlib.instruments``
   * - ``fm_violin``
     - Chowning's FM violin (vibrato, index envelope)
     - ``pysndlib.instruments``
   * - ``fm_voice``
     - vocal synthesis with vibrato (jcvoi.scm)
     - ``pysndlib.instruments``
   * - ``fofins``
     - formant synthesis (FOF) of a vowel-like voice tone
     - ``pysndlib.instruments``
   * - ``glassy``
     - glassy clang via the ``rkoddssb`` generator
     - ``pysndlib.instruments``
   * - ``gong``
     - struck, detuned gong (bank of oscillators)
     - ``pysndlib.instruments``
   * - ``gran_synth``
     - granular synthesis with control over grain duration and
       interval
     - ``pysndlib.instruments``
   * - ``hammondoid``
     - Hammond-organ-like tone (clm-ins.scm)
     - ``pysndlib.instruments``
   * - ``jc_reverb``
     - the classic Chowning/Schottstedt plate reverb (jcrev.scm);
       use as ``Sound(reverb=jc_reverb)``
     - ``pysndlib.instruments``
   * - ``jl_reverb``
     - a comb-based reverb variant (clm-ins.scm)
     - ``pysndlib.instruments``
   * - ``lbj_piano``
     - piano tones with hammer noise component (clm-ins.scm)
     - ``pysndlib.instruments``
   * - ``lutish``
     - additive tone via a cosine bank (``nrcos``)
     - ``pysndlib.instruments``
   * - ``machine1``
     - machine-like tone via FM single sideband (``fmssb``)
     - ``pysndlib.instruments``
   * - ``metal``
     - metallic struck tone from detuned oscillators
     - ``pysndlib.instruments``
   * - ``nkssber``
     - K-sine summation bank tone with vibrato (``nkssb``)
     - ``pysndlib.instruments``
   * - ``nrev``
     - the standard CLM "new" reverb (nrev); use as
       ``Sound(reverb=nrev)``
     - ``pysndlib.instruments``
   * - ``oboish``
     - oboe-like tone
     - ``pysndlib.instruments``
   * - ``organish``
     - organ-like FM tone
     - ``pysndlib.instruments``
   * - ``pianoy``, ``pianoy1``, ``pianoy2``
     - piano variants (wavetable with bounce modulation)
     - ``pysndlib.instruments``
   * - ``pluck``
     - Karplus-Strong plucked string
     - ``pysndlib.instruments``
   * - ``pqw``
     - phase-quadrature waveshaping (pqw-vox): sine and cosine
       channels driven by envelopes
     - ``pysndlib.instruments``
   * - ``resflt``
     - noise-driven resonant filter bank sweep
     - ``pysndlib.instruments``
   * - ``reson``
     - formant-filtered voice-like resonance with envelope-skewed
       formants
     - ``pysndlib.instruments``
   * - ``rhodey``
     - Fender Rhodes electric piano (STK Rhodey.cpp)
     - ``pysndlib.instruments``
   * - ``scratch``
     - turntable-style scratch of a sound file (back-and-forth
       playback between cue points)
     - ``pysndlib.instruments``
   * - ``spectra``
     - wavetable additive synthesis with amplitude-weighted partials
     - ``pysndlib.instruments``
   * - ``stereo_flute``
     - physical-model flute with noise bursts and vibrato
     - ``pysndlib.instruments``
   * - ``stringy``
     - delay-line plucked/struck string
     - ``pysndlib.instruments``
   * - ``touch_tone``
     - telephone DTMF touch tones from a digit string
     - ``pysndlib.instruments``
   * - ``tubebell``
     - tubular bell (STK TubeBell.cpp)
     - ``pysndlib.instruments``
   * - ``two_tab``
     - two wavetables crossfaded via an envelope
     - ``pysndlib.instruments``
   * - ``vox``
     - singing voice: interpolated formant tables driven by a
       phoneme list
     - ``pysndlib.clm_ins``
   * - ``wurley``
     - Wurlitzer electric piano (STK Wurley.cpp)
     - ``pysndlib.instruments``
   * - ``za``, ``zc``, ``zn``
     - pulse train through an all-pass with a time-varying length
       (feedback/feedforward variants: ``za`` all-pass, ``zc`` comb,
       ``zn`` notch)
     - ``pysndlib.instruments``

Writing your own instrument
------------------------------------

An instrument is just a function that computes its own sample loop
and writes with ``clm.outa`` and friends::

    def sine_bell(beg, dur, freq, amp):
        start = clm.seconds2samples(beg)
        length = clm.seconds2samples(dur)
        osc = clm.make_oscil(freq)
        ampe = clm.make_env([0, 0, 5, 1, 100, 0], scaler=amp, duration=dur)
        for i in range(start, start + length):
            clm.outa(i, clm.env(ampe) * clm.oscil(osc))

The notebooks include a full introduction -- writing an instrument,
using the shipped collection, and speeding up your own instruments
with cython (notebooks 04 and 05 in the repository's ``notebooks/``
directory).
