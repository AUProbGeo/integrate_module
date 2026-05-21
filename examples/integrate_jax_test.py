#!/usr/bin/env python
# %% [markdown]
# # JAX vs parallel NumPy: integrate_rejection comparison
#
# Benchmarks wall-clock time and verifies numerical agreement between the
# **parallel NumPy** backend (multiprocessing + shared memory) and the new
# **JAX** backend.  Profile plots from both backends are saved side-by-side
# so visual agreement can be confirmed.

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
import h5py
import numpy as np
import matplotlib.pyplot as plt
import integrate as ig

# %% [markdown]
# ## 1. Get example data

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, loadType='prior_data')
f_data_h5 = files[0]
f_prior_h5 = files[-1]
file_gex = ig.get_gex_file_from_data(f_data_h5)

print("Using data file:       %s" % f_data_h5)
print("Using GEX file:        %s" % file_gex)
print("Using prior data file: %s" % f_prior_h5)

# %% [markdown]
# ## 2. Setup

# %%
# Number of soundings to invert (increase for a richer profile, costs more time)
N_test = 100
ip_range = list(np.arange(N_test))

Ncpu_max = multiprocessing.cpu_count()
Ncpu_list = [c for c in [1, 2, 4, 8] if c <= Ncpu_max]
print("CPUs available: %d  (will test: %s)" % (Ncpu_max, str(Ncpu_list)))

# %% [markdown]
# ## 3. JAX warm-up
# The first JAX call triggers XLA compilation; time it separately so it does
# not inflate the benchmark numbers.

# %%
print('\nJAX warm-up (XLA compilation) ...')
t0 = time.time()
ig.integrate_rejection(
    f_prior_h5, f_data_h5,
    ip_range=[0],
    nr=50, backend='jax',
    updatePostStat=False, showInfo=-1,
)
t_warmup = time.time() - t0
print('  compile + run: %.2fs' % t_warmup)

# %% [markdown]
# ## 4. Benchmark

# %%
# --- Parallel NumPy ---
t_np = {}
f_post_np_last = None
for Ncpu in Ncpu_list:
    print('\nNumPy parallel  Ncpu=%d ...' % Ncpu)
    t0 = time.time()
    f_post_np = ig.integrate_rejection(
        f_prior_h5, f_data_h5,
        ip_range=ip_range,
        nr=200, Ncpu=Ncpu,
        backend='numpy',
        f_post_h5='POST_numpy_Ncpu%d.h5' % Ncpu,
        updatePostStat=False, showInfo=0,
    )
    t_np[Ncpu] = time.time() - t0
    f_post_np_last = f_post_np
    print('  %.2fs' % t_np[Ncpu])

# %%
# --- JAX ---
print('\nJAX  Nbatch=64 ...')
t0 = time.time()
f_post_jx = ig.integrate_rejection(
    f_prior_h5, f_data_h5,
    ip_range=ip_range,
    nr=200,
    backend='jax',
    f_post_h5='POST_jax.h5',
    updatePostStat=False, showInfo=0,
)
t_jx = time.time() - t0
print('  %.2fs' % t_jx)

# %% [markdown]
# ## 5. Timing results

# %%
t_ref = t_np[Ncpu_list[0]]

print('\n' + '='*56)
print('  Ndp=%d' % N_test)
print('='*56)
for Ncpu in Ncpu_list:
    print('  NumPy parallel  Ncpu=%-2d   %6.2fs   %.1fx' % (
        Ncpu, t_np[Ncpu], t_ref / t_np[Ncpu]))
print('  JAX  Nbatch=64            %6.2fs   %.1fx' % (t_jx, t_ref / t_jx))
print('='*56)

# %% [markdown]
# ## 6. Numerical agreement

# %%
with h5py.File(f_post_np_last, 'r') as f:
    EV_np = f['EV'][:]
    T_np  = f['T'][:]
with h5py.File(f_post_jx, 'r') as f:
    EV_jx = f['EV'][:]
    T_jx  = f['T'][:]

mask = ~np.isnan(EV_np[:N_test]) & ~np.isnan(EV_jx[:N_test])
ev_diff = np.abs(EV_np[:N_test][mask] - EV_jx[:N_test][mask])
t_diff  = np.abs(T_np[:N_test][mask]  - T_jx[:N_test][mask])

print('\nNumerical agreement (parallel NumPy vs JAX):')
print('  EV  max|diff| = %.2e   mean|diff| = %.2e' % (ev_diff.max(), ev_diff.mean()))
print('  T   max|diff| = %.2e   mean|diff| = %.2e' % (t_diff.max(),  t_diff.mean()))
if ev_diff.max() < 1e-3:
    print('\n  PASS: results agree within tolerance')
else:
    print('\n  WARN: results differ -- check implementation')

# %% [markdown]
# ## 7. Profile comparison
# Compute posterior statistics for both backends, then plot side-by-side.

# %%
print('\nComputing posterior statistics ...')
ig.integrate_posterior_stats(f_post_np_last, ip_range=ip_range)
ig.integrate_posterior_stats(f_post_jx,      ip_range=ip_range)

# %%
# NumPy profile
ig.plot_profile(f_post_np_last, im=1, i1=1, i2=N_test)
plt.suptitle('NumPy backend (Ncpu=%d)' % Ncpu_list[-1])
plt.tight_layout()
f_fig_np = 'POST_numpy_profile.png'
plt.savefig(f_fig_np, dpi=150)
print('NumPy profile saved to %s' % f_fig_np)
plt.show()

# %%
# JAX profile
ig.plot_profile(f_post_jx, im=1, i1=1, i2=N_test)
plt.suptitle('JAX backend')
plt.tight_layout()
f_fig_jx = 'POST_jax_profile.png'
plt.savefig(f_fig_jx, dpi=150)
print('JAX profile saved to %s' % f_fig_jx)
plt.show()
