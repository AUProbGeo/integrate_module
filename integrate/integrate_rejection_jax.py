"""
JAX backend for integrate_rejection likelihood calculations.

Provides JIT-compiled, vmapped Gaussian-diagonal likelihood functions and a
drop-in replacement for integrate_rejection_range that uses them for the hot
path.  Temperature estimation and posterior sampling remain in NumPy.

Key advantage over the multiprocessing backend: D is transferred to the XLA
device once per batch and reused across all data points in that batch, avoiding
the repeated memory-bandwidth competition that limits scaling past ~8 cores.

Usage
-----
    from integrate.integrate_rejection_jax import integrate_rejection_range_jax
    # or via integrate_rejection(backend='jax', ...)
"""

import numpy as np
from tqdm import tqdm

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _check_jax():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for backend='jax'.\n"
            "Install with:  pip install jax          (CPU)\n"
            "           or: pip install jax[cuda12]  (GPU)"
        )


# ---------------------------------------------------------------------------
# JAX likelihood kernels (built lazily on first use)
# ---------------------------------------------------------------------------

_single_kernel = None
_batch_kernel = None


def _get_jax_kernels():
    """Return (single, batch) JIT-compiled Gaussian-diagonal likelihood fns."""
    global _single_kernel, _batch_kernel
    if _single_kernel is not None:
        return _single_kernel, _batch_kernel

    @jax.jit
    def _likelihood_gaussian_diagonal_jax(D, d_obs, d_std):
        """
        JIT-compiled Gaussian diagonal log-likelihood for one data point.

        Parameters
        ----------
        D     : jax array (N, Nf)  — prior forward-model predictions
        d_obs : jax array (Nf,)    — observed data (may contain NaN)
        d_std : jax array (Nf,)    — per-feature standard deviation

        Returns
        -------
        jax array (N,) — log-likelihood for each prior sample
        """
        valid = ~(jnp.isnan(d_obs) | jnp.isnan(d_std))
        d_obs_s = jnp.where(valid, d_obs, 0.0)
        d_std_s = jnp.where(valid, d_std, 1.0)
        dd = D - d_obs_s
        return -0.5 * jnp.sum(valid * (dd / d_std_s) ** 2, axis=1)

    # Vectorise over a batch of data points; D is shared (in_axes=(None, 0, 0))
    _likelihood_gaussian_diagonal_batch_jax = jax.jit(
        jax.vmap(_likelihood_gaussian_diagonal_jax, in_axes=(None, 0, 0))
    )

    _single_kernel = _likelihood_gaussian_diagonal_jax
    _batch_kernel = _likelihood_gaussian_diagonal_batch_jax
    return _single_kernel, _batch_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def integrate_rejection_range_jax(
    D,
    DATA,
    idx=[],
    N_use=None,
    id_use=[],
    ip_range=[],
    nr=1000,
    autoT=1,
    T_base=1,
    T_N_above=10,
    T_P_acc_level=0.2,
    progress_callback=None,
    Nbatch=64,
    **kwargs,
):
    """
    JAX-accelerated replacement for integrate_rejection_range.

    Computes Gaussian-diagonal likelihoods for a batch of Nbatch data points
    simultaneously using a JIT-compiled, vmapped JAX kernel.  D is transferred
    to the XLA device once per data-type, then reused across all batches,
    avoiding the memory-bandwidth bottleneck of the multiprocessing backend.

    Full-covariance Gaussian and multinomial noise models fall back to the
    original NumPy implementations (unchanged behaviour).

    Parameters
    ----------
    D           : list of ndarray   — forward-modeled data per data type
    DATA        : dict              — observed data (same format as load_data)
    idx         : list              — prior sample indices (empty = all)
    N_use       : int or None       — max prior samples to evaluate
    id_use      : list              — data-type identifiers to include
    ip_range    : list              — data-point indices to invert
    nr          : int               — posterior samples to keep per data point
    autoT       : int               — 1 = auto temperature, 0 = use T_base
    T_base      : float             — base temperature when autoT=0
    T_N_above   : int               — passed to logl_T_est
    T_P_acc_level : float           — passed to logl_T_est
    progress_callback : callable    — optional (current, total) callback
    Nbatch      : int               — data points per JAX batch (default 64)
    **kwargs    : forwarded kwargs (use_N_best, showInfo, console_progress, …)

    Returns
    -------
    Same 8-tuple as integrate_rejection_range:
    (i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean,
     CHI2_all, N_UNIQUE_all, ip_range)
    """
    _check_jax()

    import integrate as ig
    from integrate.integrate_rejection import (
        likelihood_gaussian_full,
        likelihood_multinomial,
    )

    _, likelihood_gauss_diag_batch = _get_jax_kernels()

    # --- Mirror the setup from integrate_rejection_range -------------------

    use_N_best = kwargs.get('use_N_best', 0)
    showInfo = kwargs.get('showInfo', 0)
    console_progress = kwargs.get('console_progress', True)
    disableTqdm = not console_progress if showInfo >= 0 else True
    useRandomData = kwargs.get('useRandomData', True)

    Ndp = DATA['d_obs'][0].shape[0]
    if len(ip_range) == 0:
        ip_range = np.arange(Ndp)
    nump = len(ip_range)

    if len(id_use) == 0:
        Ndt = len(DATA['d_obs'])
        id_use = np.arange(Ndt)
    Ndt = len(id_use)

    noise_model = DATA['noise_model']
    i_use_data = DATA['i_use']

    # Convert multinomial class IDs to indices (same as original)
    class_is_idx = True
    class_id_list = []
    updated_data_ids = []
    for i in range(Ndt):
        if noise_model[i] == 'multinomial':
            Di, class_id, class_id_out = ig.class_id_to_idx(D[i])
            if class_is_idx and i not in updated_data_ids:
                updated_data_ids.append(i)
                D[i] = Di
            class_id_list.append(class_id_out if class_is_idx else class_id)
        else:
            class_id_list.append([])

    N = D[0].shape[0]
    if N_use is None:
        N_use = N
    N_use = min(N_use, N)
    if len(idx) == 0:
        idx = np.arange(N_use)

    # Pre-allocate output arrays
    i_use_all = np.zeros((nump, nr), dtype=np.int32)
    T_all = np.zeros(nump) * np.nan
    EV_all = np.zeros(nump) * np.nan
    EV_post_all = np.zeros(nump) * np.nan
    EV_post_all_mean = np.zeros(nump) * np.nan
    CHI2_all = np.zeros((nump, Ndt)) * np.nan
    N_UNIQUE_all = np.zeros(nump) * np.nan

    # Pre-convert D to JAX arrays for diagonal-Gaussian data types.
    # For full-covariance Gaussian and multinomial, keep None (NumPy fallback).
    use_jax_diag = [
        noise_model[i] == 'gaussian'
        and DATA['Cd'][0] is None
        and DATA['d_std'][0] is not None
        for i in range(Ndt)
    ]
    D_jax = [jnp.asarray(D[i]) if use_jax_diag[i] else None for i in range(Ndt)]

    # --- Process ip_range in batches ---------------------------------------

    for batch_start in tqdm(
        range(0, nump, Nbatch),
        disable=disableTqdm,
        desc='Rejection Sampling (JAX)',
        leave=False,
    ):
        batch_end = min(batch_start + Nbatch, nump)
        batch_js = range(batch_start, batch_end)   # indices into i_use_all / T_all / …
        ip_batch = [ip_range[j] for j in batch_js]
        bsz = len(ip_batch)

        # Accumulate log-likelihood contributions per type: (Ndt, bsz, N)
        L_per_type = np.zeros((Ndt, bsz, N))
        n_data_per_type = np.zeros((bsz, Ndt))

        for i in range(Ndt):
            active = np.array([i_use_data[i][ip] for ip in ip_batch])  # (bsz,)

            if noise_model[i] == 'gaussian':
                d_obs_batch = np.array([DATA['d_obs'][i][ip] for ip in ip_batch])  # (bsz, Nf)

                for b, ip in enumerate(ip_batch):
                    if active[b]:
                        n_data_per_type[b, i] = int(np.sum(~np.isnan(DATA['d_obs'][i][ip])))

                if DATA['Cd'][0] is not None:
                    # Full-covariance fallback: per-data-point NumPy call
                    for b, ip in enumerate(ip_batch):
                        if active[b]:
                            Cd = (DATA['Cd'][0][ip]
                                  if len(DATA['Cd'][0].shape) == 3
                                  else DATA['Cd'][0][:])
                            L_per_type[i, b] = likelihood_gaussian_full(
                                D[i], DATA['d_obs'][i][ip], Cd, N_app=use_N_best
                            )

                elif DATA['d_std'][0] is not None:
                    # Diagonal case: batched JAX kernel
                    d_std_batch = np.array([DATA['d_std'][i][ip] for ip in ip_batch])
                    L_batch = np.asarray(
                        likelihood_gauss_diag_batch(
                            D_jax[i],
                            jnp.asarray(d_obs_batch),
                            jnp.asarray(d_std_batch),
                        )
                    )  # (bsz, N)
                    # Zero out inactive data points
                    L_per_type[i] = L_batch * active[:, None]

            elif noise_model[i] == 'multinomial':
                for b, ip in enumerate(ip_batch):
                    if active[b]:
                        d_obs_ip = DATA['d_obs'][i][ip]
                        n_data_per_type[b, i] = int(np.sum(~np.isnan(d_obs_ip)))
                        L_per_type[i, b] = likelihood_multinomial(
                            D[i], d_obs_ip,
                            np.array(class_id_list[i]),
                            class_is_idx=class_is_idx,
                        )

        # Combined log-likelihood across all data types: (bsz, N)
        L_combined = np.sum(L_per_type, axis=0)

        # Per data-point: temperature estimation, sampling (NumPy, fast)
        for b, j in enumerate(batch_js):
            ip = ip_batch[b]
            L = L_combined[b]  # (N,)

            if autoT == 1:
                T = ig.logl_T_est(L, N_above=T_N_above, P_acc_lev=T_P_acc_level)
            else:
                T = T_base

            P_acc = np.exp((1.0 / T) * (L - np.nanmax(L)))
            P_acc[np.isnan(P_acc)] = 0.0

            try:
                if P_acc.shape[0] == 1:
                    P_acc = P_acc.flatten()
                p = P_acc / np.sum(P_acc)
                i_use = np.random.choice(N, nr, p=p)
            except Exception:
                i_use = np.random.choice(N, nr)

            # Reduced chi-squared per data type
            CHI2_current = np.zeros(Ndt) * np.nan
            for i in range(Ndt):
                if n_data_per_type[b, i] > 0:
                    L_acc = L_per_type[i, b, i_use]
                    CHI2_current[i] = np.nanmean(-2.0 * L_acc) / n_data_per_type[b, i]

            if useRandomData:
                i_use = idx[i_use]

            max_L = np.nanmax(L)
            EV = max_L + np.log(np.nanmean(np.exp(L - max_L)))

            i_use_all[j] = i_use
            T_all[j] = T
            EV_all[j] = EV
            EV_post_all[j] = np.nan
            EV_post_all_mean[j] = np.nan
            CHI2_all[j, :] = CHI2_current
            N_UNIQUE_all[j] = len(np.unique(i_use))

            if progress_callback is not None:
                progress_callback(j + 1, nump)

    return (
        i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean,
        CHI2_all, N_UNIQUE_all, ip_range,
    )
