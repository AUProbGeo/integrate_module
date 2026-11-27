# When anemone-core is updated

Status (2026-11-27): integrate requires `anemone-core>=0.1.1`. 0.1.1 has one
known bug that affects integrate, fixed in
[hydrogeophysicsgroup/anemone#2](https://github.com/hydrogeophysicsgroup/anemone/pull/2)
(merged 2026-11-27, not yet released on PyPI; GitHub main still says 0.1.1).

Until a release with the fix is on PyPI, `[tool.uv.sources]` in
`pyproject.toml` makes uv (`uv sync`, `uv run`, `uv lock`) install
anemone-core from GitHub at the fix commit. pip, and anyone installing
integrate-module from PyPI, still get the broken 0.1.1 from PyPI. The boxcar
xfail is removed (checklist step 4), so
`test_dual_moment_matches_single_moments[boxcar]` fails on those installs.

When the release is out: remove the `[tool.uv.sources]` entry, raise the pin
(step 3), then do steps 2 and 5-7.

Checked against anemone-core main (60ad16a), DAUGAARD line 150 vs AarhusInv,
median |rel diff|: `'boxcar'` 0.24 % (LM) / 0.20 % (HM), `'centre'` 0.77 % /
0.89 %.

## The open bug (anemone#2)

With gate integration (`t_end`), `Forward.copy()` rebuilds the Forward from
the gate *centre* times, so the copy averages dB/dt from gate centre to close
instead of open to close. `Forward.__add__` works on copies, so every
**dual-moment** system (LM + HM) is wrong with
`gate_integration='boxcar'`: -4 to -20 % for tTEM, 9 % median misfit to
AarhusInv instead of 0.2 %. The default `gate_integration='centre'` is not
affected.

Until it is released: **do not use `gate_integration='boxcar'`** with
`forward_anemone` / `prior_data_anemone`.

## Checklist for a new anemone-core release

1. Check the release includes anemone#2: see the release notes, or that
   `t_st` is a field of `BaseForward` in `anemone/forwards/forwards.py`.
2. Update the environment:
   ```bash
   uv lock --upgrade-package anemone-core
   ./uv_sync.sh
   uv pip show anemone-core          # check the new version
   ```
3. Raise the pin in `pyproject.toml`: `"anemone-core>=<new version>"`.
4. Run the anemone tests:
   ```bash
   pytest tests/test_anemone_core.py -v
   ```
   With anemone#2 in the release, `test_dual_moment_matches_single_moments[boxcar]`
   reports `XPASS(strict)` (a failure). That means the bug is fixed: remove the
   `pytest.param("boxcar", marks=pytest.mark.xfail(...))` wrapper in that
   test so `"boxcar"` is a plain parameter.
5. Consider making `gate_integration='boxcar'` the default in
   `forward_anemone`, `prior_data_anemone` and `gex_to_anemone_system`
   (`integrate/anemone_forward.py`). GA-AEM (`AreaUnderCurve`) and AarhusInv
   average over the gate window; anemone's `'centre'` samples the gate
   centre, which is 0.5-4 % low on late gates. Measured median misfit to
   AarhusInv (DAUGAARD tTEM): `'centre'` 0.8 % (LM) / 0.9 % (HM), `'boxcar'`
   (with the fix) 0.24 % / 0.20 %. This changes default results, so mention it
   in the release notes.
6. Re-run the AarhusInv comparison and check that `anemone vs AarhusInv` is
   about 0.2-0.3 % (median) with boxcar:
   ```bash
   cd examples && python integrate_compare_anemone_gaaem_aarhusinv.py
   ```
7. Run the full suite: `pytest`.

## If a release breaks something else

`integrate/anemone_forward.py` uses anemone internals in a few places, which
are the first to check:

- `_rescaled_batch` (used by `rescale_height=True`) re-implements
  `Forward.forward()`: uses `fwr.omega`, `fwr.lamda`, `fwr.H2t` (`mu0`
  already included), `fwr.primary`, `anemone.cdtype()` / `rdtype()`. A guard
  raises `RuntimeError` if it no longer reproduces `Forward` at dh=0.
- `_DampedLowPass` mimics `ButterworthFilter` (`poles()`, `stepoff_response()`).
- `_make_waveform` builds a `Waveform` via `__new__` and sets `time`,
  `amplitude`, `d2wdt2`.
- `_build_forward_uncached` sets `Forward.tmin` before `setup()` and combines
  moments with `+` (uses `t1slc` / `t2slc`).

`tests/test_anemone_core.py` covers all of these.
