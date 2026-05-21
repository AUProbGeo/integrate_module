#!/usr/bin/env python
# %% [markdown]
# # JAX backend comparison for integrate_rejection
#
# Compares the original NumPy backend against the new JAX backend:
# - timing for the likelihood hot-path
# - numerical agreement of EV (evidence) and T (temperature)
#
# Uses the DAUGAARD example dataset (downloaded if not already present).

# %%
import os, sys, time
import numpy as np
import integrate as ig
from integrate.integrate_rejection import integrate_rejection_range
from integrate.integrate_rejection_jax import integrate_rejection_range_jax

# %% [markdown]
# ## 1. Load example data

# %%
examples_dir = os.path.dirname(os.path.abspath(__file__))

f_prior_h5 = os.path.join(examples_dir, 'PRIOR_N10000_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
f_data_h5  = os.path.join(examples_dir, 'DAUGAARD_AVG.h5')

# Download if not present
if not os.path.exists(f_prior_h5) or not os.path.exists(f_data_h5):
    print('Downloading DAUGAARD example files...')
    files = ig.get_case_data(case='DAUGAARD', loadType='post')
    f_data_h5  = files[0]
    f_prior_h5 = files[3]

print(f'Data  : {f_data_h5}')
print(f'Prior : {f_prior_h5}')

# %%
DATA = ig.load_data(f_data_h5, showInfo=0)
D, idx = ig.load_prior_data(f_prior_h5, showInfo=0)

N   = D[0].shape[0]
Nf  = D[0].shape[1]
Ndp = DATA['d_obs'][0].shape[0]

# Use a fixed subset of data points so the run completes quickly
N_test = 100                          # number of soundings to invert
ip_range = np.arange(N_test)

print(f'\nPrior samples  N  = {N}')
print(f'Features       Nf = {Nf}')
print(f'Data points       = {Ndp}  (using {N_test})')

# %% [markdown]
# ## 2. JAX warm-up
# The first JAX call triggers XLA compilation; we time it separately.

# %%
kw = dict(
    D=D, DATA=DATA, idx=idx, N_use=N, id_use=[],
    ip_range=ip_range, nr=200, autoT=1,
    showInfo=-1, console_progress=False,
)

print('\nJAX warm-up (XLA compilation) ...')
t0 = time.time()
integrate_rejection_range_jax(**kw, Nbatch=64)
t_warmup = time.time() - t0
print(f'  compile + run: {t_warmup:.2f}s')

# %% [markdown]
# ## 3. Benchmark

# %%
print('\nNumPy serial ...')
t0 = time.time()
res_np = integrate_rejection_range(**kw)
t_np = time.time() - t0
print(f'  {t_np:.2f}s')

print('\nJAX (Nbatch=64) ...')
t0 = time.time()
res_jx64 = integrate_rejection_range_jax(**kw, Nbatch=64)
t_jx64 = time.time() - t0
print(f'  {t_jx64:.2f}s')

print('\nJAX (Nbatch=N_test, single batch) ...')
t0 = time.time()
res_jx_all = integrate_rejection_range_jax(**kw, Nbatch=N_test)
t_jx_all = time.time() - t0
print(f'  {t_jx_all:.2f}s')

# %% [markdown]
# ## 4. Results

# %%
print('\n' + '='*50)
print(f'  N={N}, Nf={Nf}, Ndp={N_test}')
print('='*50)
print(f'  NumPy serial          {t_np:6.2f}s   1.0x')
print(f'  JAX Nbatch=64         {t_jx64:6.2f}s   {t_np/t_jx64:.1f}x')
print(f'  JAX Nbatch={N_test}       {t_jx_all:6.2f}s   {t_np/t_jx_all:.1f}x')
print('='*50)

# %%
# Numerical agreement: EV and T are deterministic given the same L array
EV_np   = res_np[2]
T_np    = res_np[1]
EV_jx   = res_jx64[2]
T_jx    = res_jx64[1]

ev_diff = np.abs(EV_np - EV_jx)
t_diff  = np.abs(T_np  - T_jx)

print(f'\nNumerical agreement (NumPy vs JAX):')
print(f'  EV  max|diff| = {ev_diff.max():.2e}   mean|diff| = {ev_diff.mean():.2e}')
print(f'  T   max|diff| = {t_diff.max():.2e}   mean|diff| = {t_diff.mean():.2e}')

if ev_diff.max() < 1e-3:
    print('\n  PASS: results agree within tolerance')
else:
    print('\n  WARN: results differ — check implementation')
