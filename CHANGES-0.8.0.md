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

## Phase 2.5 — CI workflow modernization (`.github/workflows/build-wheels.yml`)

Context: the previous workflow was already partly broken — `upload-artifact@v3` was shut down by
GitHub (artifact uploads fail), `checkout@v3`/`setup-python@v3` are sunset, and `cibuildwheel@v2.16.5`
cannot build CPython 3.13/3.14 wheels at all. Verified against the cibuildwheel changelog:
v3.0 removed the `manylinux1` shortname (default is now `manylinux_2_28`), v3.1.0 made CPython
3.14 builds default, and the current release line is v4 (v4.3.5).

| Item | Change |
|---|---|
| Python matrix | `cp310-*` … `cp312-*` → `["cp310-*", "cp311-*", "cp312-*", "cp313-*", "cp314-*"]` on both macOS and Linux jobs (cp310 kept per user decision; it is EOL but still buildable). Free-threaded `cpNNNt` variants are excluded automatically by the per-version `cpNNN-*` glob patterns. |
| cibuildwheel | `pypa/cibuildwheel@v2.16.5` → `@v4.3.5` (current). Removed `CIBW_MANYLINUX_X86_64_IMAGE: manylinux1` / `CIBW_MANYLINUX_I686_IMAGE: manylinux1` overrides (shortname removed in cibuildwheel 3.0; cibuildwheel's default `manylinux_2_28` image is now used). |
| Actions | `checkout@v3` → `@v6`; dropped the broken `setup-python@v3` steps (they referenced an undefined `matrix.python-version`; cibuildwheel manages interpreters itself); `upload-artifact@v3` → `@v4` with unique names per job (`wheels-macos-universal2`, `wheels-linux-x86_64`, `wheels-sdist`); `download-artifact@v3` → `@v4` with `pattern: wheels-*` + `merge-multiple: true` (v4 no longer flattens multi-artifact downloads). |
| macOS job | `runs-on: macos-latest` → pinned `macos-15` (arm64). Kept `CIBW_ARCHS_MACOS: universal2` and the `libsndlib_macos_universal2.zip` release asset (arch/runner-independent). `CIBW_TEST_SKIP` narrowed from `"*-macosx_arm64 *-macosx_universal2:arm64 *-macosx_universal2:x86_64"` to `"*-macosx_universal2:x86_64"` so the arm64 slice is now **tested natively** on the arm64 runner. |
| Linux job | x86_64 only (per user decision; no aarch64 job). |
| Wheel testing | Added global `CIBW_TEST_COMMAND: "python -m unittest discover -s {project}/test"` so every wheel runs `test/test_clm.py` before upload (wheels previously shipped untested). |
| sdist job | `runs-on: macos-latest` → `macos-15`; same action updates. |
| `pyproject.toml` | Added per-version Python classifiers (3.10–3.14); replaced incorrect `"Operating System :: OS Independent"` with `MacOS :: Mac OS X` and `POSIX :: Linux`. |
| `docs/source/installation.rst` | Updated supported-versions text (3.10–3.14, previously claimed "3.10 and 3.11"). |

Workflow YAML validated for syntax. `.github/workflows/test.yml` (mostly commented-out duplicate)
was intentionally left untouched; it only runs on manual dispatch and does no harm — flagged for
a future cleanup.

## Phase 3 — Error-handler rework (`clm.pyx`)

**Behavior change:** sndlib errors now surface as `SNDLibError` raised from well-defined
checkpoints instead of as unpredictable pending CPython exceptions.

Previously `clm_error_handler` raised `SNDLibError` *from inside sndlib's C code*. sndlib never
checks Python's error indicator, so the exception stayed pending while the remaining sndlib C code
and the rest of the Cython wrapper kept executing, then surfaced at an unrelated later line
(typically as a confusing "Exception ignored" message or a bogus error at a random statement).

| Change | Detail |
|---|---|
| Store-and-raise model | `clm_error_handler` now stores the error in a module-level `_pending_error` slot (first error wins) and never raises. New public `clm.check_sndlib_errors()` raises and clears the stored error. Both handlers are declared `noexcept` and wrap their bodies in `try/except BaseException` so no exception can ever escape into sndlib C code. |
| Boundary checks | `check_sndlib_errors()` is called at: `mus_any.from_ptr` (funnel for every `make_*` factory), `out_any` (funnel for all per-sample output: `outa`/`outb`/`outc`/`outd`/`locsig` paths), `in_any`, `file2ndarray`, `ndarray2file`, and `Sound.__exit__` (only when no exception is already propagating out of the with-block, so the original error is never masked). |
| Registration order | Registration stays **after** sndlib initialization, now with a comment explaining why: sndlib's `mus_sound_initialize()` resets the error handler to its default (`sound.c: mus_error_handler = default_mus_error`), so a handler registered before init is overwritten. (The plan's original "register before init" item is not implementable against this sndlib behavior; init-time errors necessarily go to the default stderr handler.) |
| Print handler | `mus_print_set_handler` wired to a `clm_print_handler` that routes sndlib's `mus_print()` output to Python stdout (previously sndlib printed directly to the process stdout, bypassing Python's `sys.stdout`, invisible in IDEs/notebooks/redirected output). |
| GIL invariant | Documented at the handler site: all sndlib calls in pysndlib hold the GIL, so the handlers may safely touch Python objects; any future `nogil` use around sndlib calls that can trigger the handlers requires `with gil:` guards. |
| `ndarray2file` ordering | `check_sndlib_errors()` runs before the `fd < 0` fallback check, so the handler's (more informative) stored error is raised and the slot is cleared; a wrapper-raised error can never leave a stale error in the slot for a later boundary check to raise spuriously. |

**Architecture note discovered during verification:** every extension module statically links
`libsndlib.a`, so each module has its own copy of sndlib's globals — including the
error-handler slot and the sound-file cache. The Python handler registered in `clm` therefore
only sees sndlib errors from calls made through `clm`'s functions (and the instrument modules);
`pysndlib.sndlib`'s own copy keeps sndlib's default stderr handler. This matches the previous
behavior (the old raising handler was likewise registered only in `clm`'s copy) and is recorded
here as a known caveat of the static-linking design.


Verified: the smoke test triggers a real sndlib error (`mus_sound_open_input` on a non-sound file)
and asserts that `SNDLibError` surfaces via `check_sndlib_errors()` exactly once, then the slot is
clean again.



