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
