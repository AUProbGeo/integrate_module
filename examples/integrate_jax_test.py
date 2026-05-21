#!/usr/bin/env python
# %% [markdown]
# # JAX backend comparison for integrate_rejection
#
# Compares the original NumPy backend against the new JAX backend
# (backend='jax') for the likelihood hot-path in integrate_rejection.
#
# The JAX backend uses a JIT-compiled, vmapped kernel that processes a batch
# of data points simultaneously, reading D once per batch instead of once per
# core — avoiding the cache-thrashing that limits the multiprocessing backend
# past ~8 CPUs.
#
# Results shown:
# - Wall-clock time for NumPy serial vs JAX (two batch sizes)
# - Numerical agreement of EV (evidence) and T (temperature)

# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

# %%
import time
import numpy as np
import integrate as ig
from integrate.integrate_rejection import integrate_rejection_range
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

# Number of soundings to invert — keep modest so the example runs in < 1 min
N_test = 500
ip_range = np.arange(N_test)

print("Prior samples  N  = %d" % N)
print("Features       Nf = %d" % Nf)
print("Data points       = %d  (using %d)" % (Ndp, N_test))

# %% [markdown]
# ## 3. JAX warm-up
# The very first JAX call triggers XLA compilation; we time it separately so
# it does not inflate the benchmark.

# %%
kw = dict(
    D=D, DATA=DATA, idx=idx, N_use=N, id_use=[],
    ip_range=ip_range, nr=200, autoT=1,
    showInfo=-1, console_progress=True,
)

print('\nJAX warm-up (XLA compilation) ...')
t0 = time.time()
integrate_rejection_range_jax(**kw, Nbatch=64)
t_warmup = time.time() - t0
print('  compile + run: %.2fs' % t_warmup)

# %% [markdown]
# ## 4. Benchmark

# %%
print('\nNumPy serial ...')
t0 = time.time()
res_np = integrate_rejection_range(**kw)
t_np = time.time() - t0
print('  %.2fs' % t_np)

print('\nJAX  Nbatch=64 ...')
t0 = time.time()
res_jx64 = integrate_rejection_range_jax(**kw, Nbatch=64)
t_jx64 = time.time() - t0
print('  %.2fs' % t_jx64)

print('\nJAX  Nbatch=%d (single batch) ...' % N_test)
t0 = time.time()
res_jx_all = integrate_rejection_range_jax(**kw, Nbatch=N_test)
t_jx_all = time.time() - t0
print('  %.2fs' % t_jx_all)

# %% [markdown]
# ## 5. Results

# %%
print('\n' + '='*52)
print('  N=%d, Nf=%d, Ndp=%d' % (N, Nf, N_test))
print('='*52)
print('  NumPy serial              %6.2fs   1.0x' % t_np)
print('  JAX  Nbatch=64            %6.2fs   %.1fx' % (t_jx64,  t_np / t_jx64))
print('  JAX  Nbatch=%-4d          %6.2fs   %.1fx' % (N_test, t_jx_all, t_np / t_jx_all))
print('='*52)

# %%
# Numerical agreement: EV and T are deterministic given the same L values
EV_np = res_np[2][ip_range]
T_np  = res_np[1][ip_range]
EV_jx = res_jx64[2]
T_jx  = res_jx64[1]

ev_diff = np.abs(EV_np - EV_jx)
t_diff  = np.abs(T_np  - T_jx)

print('\nNumerical agreement (NumPy vs JAX):')
print('  EV  max|diff| = %.2e   mean|diff| = %.2e' % (ev_diff.max(), ev_diff.mean()))
print('  T   max|diff| = %.2e   mean|diff| = %.2e' % (t_diff.max(),  t_diff.mean()))

if ev_diff.max() < 1e-3:
    print('\n  PASS: results agree within tolerance')
else:
    print('\n  WARN: results differ -- check implementation')
