# pysndlib notebooks

A hands-on tour of the library, following the original CLM documentation
([sndclm.html](https://ccrma.stanford.edu/software/snd/snd/sndclm.html),
[fm.html](https://ccrma.stanford.edu/software/snd/snd/fm.html)) and the
[pysndlib docs](https://testcase.github.io/pysndlib/index.html).

Run them in order with `jupyter notebook` (requires an installed pysndlib,
plus `numpy` and `matplotlib`):

| # | Notebook | Topics |
|---|----------|--------|
| 01 | [Basics](01_Basic.ipynb) | first sound, reading it back |
| 02 | [Generators and Envelopes](02_Generators_and_Envelopes.ipynb) | `Sound`, oscil, env, FM input, waveform alternatives, noise |
| 03 | [Filters, Delays and DSP](03_Filters_Delay_DSP.ipynb) | one-pole, formant banks, delay lines, the `dsp` module |
| 04 | [Instruments](04_Instruments.ipynb) | writing instruments, fm_violin, the shipped collection, reverb |
| 05 | [Speed with Cython](05_Speed_with_Cython.ipynb) | `%%cython`, `@cython.ccall`, benchmarks |
| 06 | [FM Theory](06_FM_Theory.ipynb) | Bessel functions, sidebands, index, carrier:modulator ratios |
| 07 | [FM Instruments](07_FM_Instruments.ipynb) | fm_bell, fm_drum, fm_trumpet, fm_violin |
| 08 | [Advanced FM](08_FM_Advanced.ipynb) | composite modulators, feedback, asymmetric_fm, ssb_am |
| 09 | [Waveforms](09_Waveforms.ipynb) | the generator zoo: polynomial, band-limited, wavetables |
| 10 | [Noise](10_Noise.ipynb) | rand/rand_interp, pink/brown noise, smoothing generators |
| 11 | [Delays and Physical Modeling](11_Delays_Physical.ipynb) | comb resonance, strings, reverb internals |
| 12 | [Granulate](12_Granulate.ipynb) | grain parameters, expansion, live-function inputs |
| 13 | [Phase Vocoder](13_Phase_Vocoder.ipynb) | pitch without time, frame sizes, the time-stretch recipe |
| 14 | [Spectral Tools](14_Spectral_Tools.ipynb) | fft windows, mus_fft, polar form, spectrograms |
| 15 | [Sound Files](15_Sound_Files.ipynb) | mus_sound_* API, headers, formats, round-trips |
| 16 | [Spatialization and Reverb](16_Spatialization_Reverb.ipynb) | locsig, moving sources, reverb instruments |

Notebooks are committed without outputs; each one is runnable top-to-bottom.
