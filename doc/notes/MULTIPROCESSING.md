# Multiprocessing in INTEGRATE

This document explains how the INTEGRATE module ensures parallel processing works
transparently on Windows, macOS, and Linux without requiring users to wrap their
scripts in `if __name__ == '__main__':` guards.

---

## Background: Why multiprocessing is hard on Windows and macOS

Python's `multiprocessing` module supports three ways to start worker processes:

| Method | Platforms | How workers are created |
|--------|-----------|------------------------|
| `fork` | Linux only | Copy of parent process memory (fast, safe on Linux) |
| `spawn` | Windows, macOS, Linux | Fresh Python interpreter re-imports everything |
| `forkserver` | macOS, Linux | Dedicated server process handles forking |

On **Linux**, `fork` is used. Worker processes inherit the parent's full memory
image and immediately execute only the assigned task function — the user's script
is never re-run.

On **Windows** and **macOS** (Python 3.8+), `fork` is unavailable or deprecated.
Python uses `spawn`, which starts a fresh Python interpreter for every worker.
The spawn mechanism does two things:

1. Imports all modules the worker needs.
2. **Re-executes the user's `__main__` script** to reconstruct the global namespace.

Step 2 is the problem. Without protection, every worker would re-run the entire
user script — creating prior files, copying HDF5 files, running inversions — in
parallel, causing file lock collisions and incorrect results.

The standard Python solution is to guard all top-level code with
`if __name__ == '__main__':`. INTEGRATE avoids this requirement by handling it
internally.

---

## Solution: `__main__.__spec__` patching

### How Python's spawn bootstrap decides what to re-run

When the spawn pool is about to create workers, Python collects preparation data
via `multiprocessing.spawn.get_preparation_data()`. This function inspects
`sys.modules['__main__']`:

- If `__main__.__spec__` is **`None`** (a plain script run with `python script.py`):
  sets `init_main_from_path = script_path` → workers call
  `runpy.run_path(script_path, run_name='__main__', ...)` → **user script is
  re-executed in every worker**.

- If `__main__.__spec__` is **not `None`** (a module run with `python -m pkg.mod`):
  sets `init_main_from_name = spec.name` → workers call
  `_fixup_main_from_name(name)`.

### The trick

`_fixup_main_from_name(name)` begins with:

```python
def _fixup_main_from_name(name):
    main_module = sys.modules['__main__']
    if main_module.__name__ == name:
        return          # already correct — nothing to do
    ...
```

In a freshly spawned worker, `sys.modules['__main__'].__name__` is already
`'__main__'` (set by the bootstrap entry-point). So if we can make
`init_main_from_name = '__main__'`, the function returns immediately and the
user's script is **never re-executed**.

We achieve this by temporarily setting:

```python
sys.modules['__main__'].__spec__ = types.SimpleNamespace(name='__main__')
```

immediately before creating the Pool. `get_preparation_data()` then includes
`init_main_from_name = '__main__'` in the preparation data, and workers skip
`runpy.run_path` entirely.

After the Pool exits (guaranteed by `try/finally`), `__spec__` is restored:

```python
_main_module.__spec__ = None
```

### Complete pattern used in both Pool-creating functions

```python
import sys
import types

_main_module = sys.modules.get('__main__')
_spec_patched = _main_module is not None and getattr(_main_module, '__spec__', None) is None
if _spec_patched:
    _main_module.__spec__ = types.SimpleNamespace(name='__main__')

try:
    with ctx.Pool(processes=Ncpu) as p:
        results = p.map(worker_func, tasks)
finally:
    if _spec_patched:
        _main_module.__spec__ = None
```

The condition `getattr(_main_module, '__spec__', None) is None` ensures we only
patch when running as a plain script. When the user runs `python -m mypackage`,
`__spec__` is already set correctly and is left untouched.

---

## Spawn context choices

Both Pool-creating functions use explicit contexts rather than the global default:

### `prior_data_gaaem` (`integrate/integrate.py`)

```python
is_spawn = os.name == 'nt' or (os.name == 'posix' and os.uname().sysname == 'Darwin')
if is_spawn:
    if os.name == 'nt':
        Ncpu = min(Ncpu, 60)   # stay below Windows handle limit of ~63
    ctx = multiprocessing.get_context('spawn')
else:
    ctx = multiprocessing.get_context('fork')
```

The worker function `forward_gaaem_chunk` calls the `gatdaem1d` C extension
(a compiled DLL). This DLL has internal global state and is not thread-safe, so
processes (not threads) are required. Spawn is the only process-creation method
available on Windows; it is also used on macOS to avoid Apple's deprecated `fork`
restriction.

### `integrate_posterior_main` (`integrate/integrate_rejection.py`)

```python
if os.name == 'nt' or (os.name == 'posix' and os.uname().sysname == 'Darwin'):
    ctx = multiprocessing.get_context('spawn')
else:
    ctx = multiprocessing.get_context('fork')
```

Large prior arrays (`D`) are passed to workers via **shared memory**
(`multiprocessing.shared_memory`), so spawn overhead is minimal — no large
array serialization occurs.

---

## Safety guards on individual functions

As a belt-and-suspenders fallback, key user-facing functions also check whether
they are running inside a worker process and return `None` immediately if so:

```python
if multiprocessing.current_process().name != 'MainProcess':
    return None
```

Pool workers are always named `SpawnPoolWorker-N`, `ForkPoolWorker-N`, etc. —
never `'MainProcess'`. This guard therefore correctly identifies any worker
context on all platforms without requiring any setup or cleanup.

Functions that carry this guard:

| Function | File |
|----------|------|
| `prior_data_gaaem()` | `integrate/integrate.py` |
| `prior_model_layered()` | `integrate/integrate.py` |
| `prior_model_workbench()` | `integrate/integrate.py` |
| `prior_model_workbench_direct()` | `integrate/integrate.py` |
| `integrate_rejection()` | `integrate/integrate_rejection.py` |

---

## `multiprocessing.freeze_support()`

```python
multiprocessing.freeze_support()
```

is called at the top of `integrate/__init__.py`. This is required for frozen
executables (PyInstaller, cx_Freeze) on Windows, where the worker bootstrap
mechanism differs from normal Python. It is a no-op during regular script
execution.

---

## Stale HDF5 file handling

When a previous run crashes mid-write, it can leave a partially-written or
locked HDF5 file on disk. On Windows/WSL NTFS mounts, `h5py.File(path, 'w')`
can fail with an I/O error if the existing file is in an inconsistent state.

`copy_hdf5_file()` in `integrate_io.py` explicitly removes the output file
before opening it for writing:

```python
if os.path.exists(output_filename):
    try:
        os.remove(output_filename)
    except OSError:
        pass
output_file = h5py.File(output_filename, 'w')
```

---

## Summary: what users need to do

**Nothing.** On all platforms (Windows, macOS, Linux), user scripts can call
INTEGRATE functions at the top level without any `if __name__ == '__main__':`
guard:

```python
import integrate as ig

f_prior_h5 = ig.prior_model_layered(N=10000, ...)
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5)
```

This works correctly whether run from:
- Command line: `python my_script.py`
- IPython / Jupyter: `%run my_script.py` or notebook cells
- IDE run buttons (VS Code, PyCharm, Spyder)
- Module invocation: `python -m mypackage` (in this case `__spec__` is already
  set correctly by Python and is left untouched)
