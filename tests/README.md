# Tests for integrate_module

The test suite uses [pytest](https://docs.pytest.org). Configuration lives in
`[tool.pytest.ini_options]` in `pyproject.toml`.

The first tests check that `ig.integrate_rejection` gives the **same results**
with the `numpy` and `jax` backends.

## Running the tests

Install pytest (part of the `dev` extra) once:

```bash
./uv_sync.sh                       # full dev environment incl. jax-cuda, anemone, ga-aem
# or, to only add the dev extra without touching anything else:
uv sync --inexact --extra dev      # or: pip install -e ".[dev]"
```

Do **not** run plain `uv sync --extra dev`: `uv sync` removes every package
that is not in the requested extras, which uninstalls the JAX CUDA plugin
(`jax-cuda` extra) and the locally installed `anemone` / `ga-aem` packages.

Run from the repository root:

```bash
# all CPU tests (GPU tests are skipped automatically)
pytest

# verbose, show each test
pytest -v

# one file / one test / one parametrization
pytest tests/test_likelihood_kernels.py
pytest tests/test_rejection_backends.py::test_evidence_identical
pytest "tests/test_rejection_backends.py::test_evidence_identical[gauss_fullcd-autoT]"

# select by name substring
pytest -k evidence
pytest -k "multinomial and not x64"

# GPU tests (needs a JAX GPU device)
JAX_PLATFORMS=cuda pytest -m gpu

# everything except slow tests
pytest -m "not slow"

# stop at first failure, show local variables
pytest -x -l
```

Using `uv` without activating the venv: `uv run pytest -v`.

The default CPU run takes a few seconds.

## Layout

| File | Purpose |
|------|---------|
| `conftest.py` | Environment defaults, skip logic for `jax`/`gpu` markers, shared fixtures and the `run_rejection()` helper |
| `cases.py` | Small synthetic prior/data cases (the `CASES` registry) |
| `test_rejection_backends.py` | numpy vs JAX backend of `integrate_rejection` |
| `test_likelihood_kernels.py` | JAX kernels vs their NumPy counterparts (unit level) |

### Environment defaults (`conftest.py`)

Set before `integrate` is imported, unless already set in the shell:

- `MPLBACKEND=Agg`: no plot windows.
- `JAX_PLATFORMS=cpu`: JAX runs on CPU by default. On CPU the JAX backend
  draws posterior samples with the same `np.random` stream as the numpy
  backend, so results can be compared sample by sample. Set
  `JAX_PLATFORMS=cuda` to run on the GPU.

### Markers

| Marker | Meaning |
|--------|---------|
| `jax` | needs `jax`; skipped if it is not installed |
| `gpu` | needs a JAX GPU device; skipped when JAX runs on CPU |
| `slow` | larger cases (none yet); deselect with `-m "not slow"` |

### Fixtures and helpers

- `case`: parametrized over every case in `cases.CASES`. A test that takes
  `case` runs once per case.
- `build_case(name)`: builds one named case. Each case is built once per
  session and cached.
- `jax_x64`: turns on JAX 64-bit mode for one test and restores it afterwards.
- `run_rejection(case, backend, out_dir, seed=1, nr=400, **kwargs)`: seeds
  `np.random`, runs `ig.integrate_rejection` with `Ncpu=1, parallel=False`
  (the multiprocessing path has per-worker RNG state and cannot be reproduced),
  and returns a dict with `i_use, T, EV, CHI2, N_UNIQUE` and, with
  `updatePostStat=True`, the posterior statistics (`M1_Mean`, `M1_Std`, ...).

### Synthetic cases (`cases.py`)

The cases use a linear forward model (`D = M @ G`), so they need no EM
forward code and build in well under a second. Defaults: 2000 prior
realizations, 10 model parameters, 15 channels, 25 data points.

| Case | Exercises |
|------|-----------|
| `gauss_diag` | diagonal Gaussian noise (JAX on-device kernel) |
| `gauss_em_scaled` | EM-like magnitudes (data 1e-8 to 1e-12, 3% noise), large normalization constant |
| `gauss_diag_nan` | NaN channels in `d_obs` |
| `gauss_fullcd` | full covariance `Cd` (NumPy fallback inside the JAX backend) |
| `multinomial` | discrete model with multinomial data |
| `multi_type_masked` | two data types, the second switched off (`i_use=0`) at half the points |

## What "same results" means

- **Deterministic outputs:** the evidence `EV` and temperature `T` depend
  only on the likelihood, so they must agree on every backend and device:
  - default float32 JAX: `EV` to `rtol=1e-6`, `T` to `rtol=1e-5`
  - JAX x64 mode: `rtol=1e-12`
- **Sampled outputs:** `i_use`, `N_UNIQUE`, `CHI2` and posterior statistics
  depend on the random draws.
  - CPU: identical for the same seed.
  - GPU: JAX uses `jax.random`, so only `EV`/`T` are compared exactly and the
    posterior mean is compared within Monte-Carlo error.

### Known issues (strict xfail)

`KNOWN_X64_ISSUES` in `test_rejection_backends.py` lists cases where one
backend still computes in float32 when JAX x64 mode is on. For these cases
`EV` agrees only to float32 precision. The xfail is `strict=True`, so once a
cause is fixed the test reports XPASS and fails, as a reminder to remove the
entry.

## Adding tests

**A new synthetic case:** add a builder to `cases.py` and register it:

```python
def build_my_case(f_prior_h5, f_data_h5, rng):
    M = _prior_model(rng)
    D = M @ rng.normal(size=(NM, ND))
    _save_prior(f_prior_h5, [M], [D])
    d_obs, i_true = _observed(D, rng)
    _save_gauss(f_data_h5, d_obs, 1, D_std=np.full_like(d_obs, NOISE_STD))
    return 1, {'i_true': i_true}       # (number of data types, info dict)

CASES['my_case'] = build_my_case
```

Every test that uses the `case` fixture then also runs on `my_case`.

**A new test area:** create `tests/test_<area>.py` and reuse the fixtures:

```python
import numpy as np
from conftest import run_rejection

def test_something(build_case, tmp_path):
    case = build_case("gauss_diag")
    r = run_rejection(case, "numpy", tmp_path, updatePostStat=True)
    assert np.all(np.isfinite(r["EV"]))
```

Tests that need an optional package should call
`pytest.importorskip("<package>")` (for example for the `ga-aem`, `anemone`
or `simpeg` forward backends) and be marked `slow` if they take more than a
few seconds.
