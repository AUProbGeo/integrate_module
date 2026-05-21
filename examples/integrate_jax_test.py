#!/usr/bin/env python
# %% [markdown]
# # JAX backend comparison for integrate_rejection
#
# Compares the **parallel NumPy** backend (multiprocessing + shared memory)
# against the new **JAX** backend for the likelihood hot-path.
#
# The JAX backend uses a JIT-compiled, vmapped kernel that reads D once per
# batch of data points instead of once per CPU core, avoiding the
# memory-bandwidth contention that limits the multiprocessing path past ~8 CPUs.
#
# Results shown:
# - Wall-clock time: parallel NumPy (several Ncpu values) vs JAX
# - Numerical agreement of EV (evidence) and T (temperature)

# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

# %%
import multiprocessing
import time
import numpy as np
import integrate as ig
from integrate.integrate_rejection import integrate_posterior_main
from integrate.integrate_rejection_jax import integrate_rejection_range_jax

# %% [markdown]
# ## 1. Get example data
# Downloads the DAUGAARD case (data + forward-modelled prior) if not already present.

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, loadType='prior_data')
f_data_h5 = files[0]
f_prior_h5 = files[-1]
file_gex = ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)
print("Using prior model and data file: %s" % f_prior_h5)

# %% [markdown]
# ## 2. Load prior and observed data

# %%
DATA = ig.load_data(f_data_h5, showInfo=0)
D, idx = ig.load_prior_data(f_prior_h5, showInfo=0)

N   = D[0].shape[0]
Nf  = D[0].shape[1]
Ndp = DATA['d_obs'][0].shape[0]

# Number of soundings to invert
N_test = 10
ip_range = np.arange(N_test)

print("Prior samples  N  = %d" % N)
print("Features       Nf = %d" % Nf)
print("Data points       = %d  (using %d)" % (Ndp, N_test))

# CPU count for parallel NumPy runs
Ncpu_max = multiprocessing.cpu_count()
Ncpu_list = [c for c in [1, 2, 4, 8] if c <= Ncpu_max]
print("CPUs available: %d  (will test: %s)" % (Ncpu_max, str(Ncpu_list)))

# %% [markdown]
# ## 3. JAX warm-up
# The very first JAX call triggers XLA compilation; time it separately so it
# does not inflate the benchmark.

# %%
kw_jax = dict(
    D=D, DATA=DATA, idx=idx, N_use=N, id_use=[],
    ip_range=ip_range, nr=200, autoT=1,
    showInfo=-1, console_progress=False,
)

print('\nJAX warm-up (XLA compilation) ...')
t0 = time.time()
integrate_rejection_range_jax(**kw_jax, Nbatch=64)
t_warmup = time.time() - t0
print('  compile + run: %.2fs' % t_warmup)

# %% [markdown]
# ## 4. Benchmark

# %%
# --- Parallel NumPy ---
t_np = {}
res_np_last = None
for Ncpu in Ncpu_list:
    ip_range_shuffled = ip_range.copy()
    np.random.shuffle(ip_range_shuffled)
    ip_chunks = np.array_split(ip_range_shuffled, Ncpu)

    print('\nNumPy parallel  Ncpu=%d ...' % Ncpu)
    t0 = time.time()
    res_np = integrate_posterior_main(
        ip_chunks=ip_chunks,
        D=D, DATA=DATA, idx=idx, N_use=N, id_use=[],
        autoT=1, T_base=1, nr=200, Ncpu=Ncpu,
        use_N_best=0, T_N_above=10, T_P_acc_level=0.2,
    )
    t_np[Ncpu] = time.time() - t0
    res_np_last = res_np
    print('  %.2fs' % t_np[Ncpu])

# %%
# --- JAX ---
print('\nJAX  Nbatch=64 ...')
t0 = time.time()
res_jx = integrate_rejection_range_jax(**kw_jax, Nbatch=64)
t_jx = time.time() - t0
print('  %.2fs' % t_jx)

# %% [markdown]
# ## 5. Results

# %%
t_ref = t_np[Ncpu_list[0]]   # single-CPU parallel NumPy as reference

print('\n' + '='*56)
print('  N=%d, Nf=%d, Ndp=%d' % (N, Nf, N_test))
print('='*56)
for Ncpu in Ncpu_list:
    print('  NumPy parallel  Ncpu=%-2d   %6.2fs   %.1fx' % (
        Ncpu, t_np[Ncpu], t_ref / t_np[Ncpu]))
print('  JAX  Nbatch=64            %6.2fs   %.1fx' % (t_jx, t_ref / t_jx))
print('='*56)

# %%
# Numerical agreement between JAX and parallel NumPy (Ncpu_list[-1])
EV_np = res_np_last[2]
T_np  = res_np_last[1]
EV_jx = np.zeros(Ndp) * np.nan
T_jx  = np.zeros(Ndp) * np.nan
EV_jx[ip_range] = res_jx[2]
T_jx[ip_range]  = res_jx[1]

ev_diff = np.abs(EV_np[ip_range] - EV_jx[ip_range])
t_diff  = np.abs(T_np[ip_range]  - T_jx[ip_range])

print('\nNumerical agreement (parallel NumPy vs JAX):')
print('  EV  max|diff| = %.2e   mean|diff| = %.2e' % (ev_diff.max(), ev_diff.mean()))
print('  T   max|diff| = %.2e   mean|diff| = %.2e' % (t_diff.max(),  t_diff.mean()))

if ev_diff.max() < 1e-3:
    print('\n  PASS: results agree within tolerance')
else:
    print('\n  WARN: results differ -- check implementation')
