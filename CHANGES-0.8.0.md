# pysndlib 0.8.0 change record

Baseline: commit `64ea37f` ("Update conf.py").

## Phase 1 — Cython/NumPy compatibility & hygiene

All changes verified with: clean `cython` compile (zero warnings, previously 13
`c_string_encoding` warnings), `pip wheel` build under Cython 3.3.1 / NumPy 2.5.4 /
Python 3.12 (zero compiler warnings), smoke render (`make_oscil` → `outa` →
`file2ndarray`), and `test/test_clm.py` (3/3 OK).

| File | Change |
|---|---|
| `src/pysndlib/clm.pyx` | Added `np.import_array()` (module previously never initialized the numpy C-API; latent segfault on any future `PyArray_*` use). Added `import numpy.typing as npt` and `from typing import Optional` (annotations referenced them without import). Added `language_level=3` directive; moved `c_string_encoding` directive off line 2 (silences Python-vs-Cython encoding-declaration warning). |
| `src/pysndlib/sndlib.pyx` | Added `np.import_array()`. Restored the direct `cimport pysndlib.csndlib as csndlib` (was commented out; symbol previously resolved only indirectly via `sndlib.pxd`). Added/moved directives as above. |
| `src/pysndlib/env.pyx` | Added `np.import_array()`, `language_level=3` directive. |
| `src/pysndlib/generators.pyx` | Added `np.import_array()`, `language_level=3` directive. |
| `src/pysndlib/dsp.pyx` | Added `language_level=3` directive. (No numpy cimport in this module; no `import_array` needed.) |
| `src/pysndlib/analog_filter.pyx` | Added `language_level=3` directive. |
| `src/pysndlib/clm_ins.pyx` | Added `language_level=3` directive. (Already had correct `np.import_array()`.) |
| `src/pysndlib/clm.pxd` | Removed dead `np.import_array()` statement (Cython ignores executable statements in `.pxd` files). Header directive reorder. |
| `src/pysndlib/generators.pxd` | Removed dead `np.import_array()` statement. Header directive reorder. |
| `src/pysndlib/sndlib.pxd` | Removed dead `csndlib.mus_sound_initialize()` statement (real call is `sndlib.pyx:15`). Fixed malformed `size: int` annotation in `mus_sound_override_header` declaration. Header directive reorder. |
| `src/pysndlib/csndlib.pxd` | Replaced `ctypedef signed long int64_t` shadow typedef with `from libc.stdint cimport int64_t, uint8_t` (correct on LLP64 too; no-op on macOS/Linux). |
| `src/pysndlib/cclm.pxd` | Same `int64_t`/`uint64_t` fix via `from libc.stdint cimport int64_t, uint64_t`. |
| `src/pysndlib/dsp.pxd` | **Deleted.** Its `cpdef` declarations conflicted with the `@cython.ccall`-decorated definitions in `dsp.pyx` (known source of signature-mismatch errors between Cython minor versions); nothing cimports `pysndlib.dsp`, so the duplicate interface declarations were removed. |
| `setup.py` | Replaced deprecated `Cython.Compiler.Options.annotate = False` global poke with `cythonize(..., annotate=False)`. |
| `requirements.txt` | Updated stale pins `Cython==3.0.5`, `numpy==1.25.1` → `Cython>=3.0.5`, `numpy>=1.25.1` (matching `pyproject.toml`). |

## Phase 2 — Bug-fix sweep

**API changes in this phase** (recorded for release notes):
- `pysndlib.sndlib.mus_sound_reopen_output(file, chans, sample_type, header_type, location)` — the
  `srate` parameter was removed: it was never passed to the C function (which has no srate
  parameter) and silently did nothing.
- `pysndlib.generators.make_nchoosekcos(frequency, ratio, n)` — the previous signature
  `make_nchoosekcos(frequency, n, amplitude)` passed `n` into the constructor's `ratio` slot and
  `amplitude` (an attribute the class does not have) into `n`; the factory now matches the
  `Nchoosekcos` constructor.
- `pysndlib.clm.locsig()` now returns `None` (declared `void`); previously declared
  `cpdef double` with no return statement, so it silently returned `0.0`. All in-repo callers use
  it as a statement.

| File | Change |
|---|---|
| `src/pysndlib/sndlib.pyx` | `mus_sound_override_header`: passed the **type object `str`** instead of the `file` argument (TypeError on every call). `mus_sound_loop_info`: copied sndlib's calloc'd buffer into an owned `np.intc` array and freed the original (was a zero-copy view over leaked memory — leak on every call). `mus_sound_reopen_output`: removed unused `srate` param. Replaced now-unused `from cython cimport view` with `from libc.stdlib cimport free`. |
| `src/pysndlib/sndlib.pxd` | `mus_sound_reopen_output` declaration updated to match. |
| `src/pysndlib/dsp.pyx` | `biquad()`: added missing `return` (silently returned 0.0). `down_oct()`: returned filename (`_down_oct.wav`) that didn't match the file written (`down_oct.wav`). `freqdiv()`: moved `outname is None` validation out of the `n > 1` branch (previously reachable crash for `n <= 1`) and added `return outname`; removed dead typed local `i`. |
| `src/pysndlib/clm.pyx` | Deleted broken `SNDLibError.___init___` (triple-underscore, never executed, referenced nonexistent `self.message`). `validate_envelope`: fixed inverted `all(...)` → `not all(...)` so non-numeric envelopes are rejected; fixed missing `f` prefix on the even-length error message; `isinstance(i, int \| float)` → `isinstance(i, (int, float))`. `is_odd`/`is_even` were swapped. Removed debug `print('ndim', arr.ndim)` from `ndarray2file`. `ndarray2file`: added `fd < 0` failure check on `mus_sound_open_output`; restructured `try/finally` so the channel-pointer fill loop and the write happen in `try` (previously a failed fill loop wrote uninitialized malloc pointers and the write/close ran in `finally`, masking the original exception). `make_two_pole`: was passing `(a0, b2, b2)` — `b1` never reached sndlib. `make_fir_coeffs`: `np.ndarray(envelope)` → `np.array(envelope)`. `partials2wave`: `if not wave:` → `if wave is None:` (ambiguous truth test raised `ValueError` for multi-element ndarrays). `make_granulate` (callable-input branch): passed `NULL` as the `void*` user-arg instead of casting a Python callable to `<mus_any>` (AttributeError at runtime). `in_any`: tested the builtin `input` instead of the parameter `inp`, making the ndarray branch unreachable. `locsig`: declared `void` (was `cpdef double` with no return). `Sound.__enter__`: `self.filename` (never set; AttributeError with `continue_old_file=True`) → `self.output`; saves `default.output`/`default.reverb` for restoration. `Sound.__exit__`: restores `default.output`/`default.reverb` after `finalize`; `np.int_` → `np.int64` for the `mus_long_t[:]` buffers. `mus_any.from_ptr`: raises `MemoryError` when the `mus_make_*` pointer is NULL (previously operated on NULL). |
| `src/pysndlib/clm.pxd` | `locsig` declaration updated to `void`. |
| `src/pysndlib/generators.pyx` | `RoundInterp.mus_phase` setter: targeted nonexistent `self.gen.rnd` → `self.rnd.mus_phase`. `Nchoosekcos.__init__`: initializes `self.angle` (declared `cdef public double` in the .pxd — previously read uninitialized memory). `make_nchoosekcos`: argument order fixed (see API changes). `SincTrain.__init__`: removed `self.n = n` line that overwrote the widened `self.n = 1 + 2*n` used by `next()`'s normalization. `MovingFFT.__init__`: removed dead `isinstance(self.last_moving_fft_window, np.ndarray)` check (the attribute was set to `False` immediately before). `MovingFFT.mus_run`: returns `self.next()` (previously discarded the result and returned `None`). |
| `setup.py` | Rewritten: the three copy-pasted per-platform extension lists are now a single list; platform differences are computed once and attached via per-Extension `extra_compile_args`/`extra_link_args` instead of mutating `CFLAGS`/`LDFLAGS` environment variables (which previously overwrote any user-set values and leaked into every subprocess). `pysndlib.clm_ins` and `pysndlib.analog_filter` added to the build (previously never compiled; per user decision). |

**Deviations from plan:** the `setup.py` deduplication (originally Phase 5) was pulled into this
phase because the two newly built extensions would otherwise have had to be added to all three
platform lists. The `MovingFFT.next()` return type was left as `cython.double` (bool flag coerced
to 1.0/0.0) because the base class `CLMGenerator.next` contract is `double`; changing it to `bint`
made the override signature incompatible.

