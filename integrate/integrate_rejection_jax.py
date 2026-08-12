"""
JAX backend for integrate_rejection likelihood calculations.

All computation — likelihood evaluation AND post-processing (temperature
estimation, weighted sampling, evidence, CHI2) — is performed on-device.
Only the tiny final arrays are transferred back to the host:

    Old:  (bsz, N) likelihood matrix → ~256 MB per batch  (N=1 M, bsz=64)
    New:  i_use + scalars            → ~52 KB  per batch  (~5000× less)

This eliminates the PCIe bottleneck that made the old GPU path ~49× slower
than CPU despite the faster kernel.

Usage
-----
    from integrate.integrate_rejection_jax import integrate_rejection_range_jax
    # or via integrate_rejection(backend='jax', ...)
"""

import os
import functools
import numpy as np
from tqdm import tqdm

# Disable JAX's default behaviour of pre-allocating ~75 % of GPU VRAM upfront.
# Without this, JAX tries to grab ~18 GB on a 24 GB card at import time, which
# fails when the display driver or other processes already occupy some VRAM.
# Must be set before `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax
    import jax.numpy as jnp
    # Cache compiled GPU kernels to disk.  GPU kernel compilation for large
    # static shapes (N=1M sort, cumsum, searchsorted) takes ~40s on first run.
    # With the cache, every subsequent run reloads compiled kernels and warmup
    # drops to ~1s.  Must be set via jax.config.update (not an env var) in
    # JAX 0.10+; the cache is keyed on kernel + GPU arch so it is safe to share.
    _cache_dir = os.path.expanduser("~/.cache/jax_xla_gpu")
    os.makedirs(_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
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
# JAX post-processing kernels — temperature, sampling, EV, CHI2 on-device
# ---------------------------------------------------------------------------

def _logl_T_est_jax(L, N_above, P_acc_lev):
    """
    JAX port of integrate.logl_T_est.

    Estimates an annealing temperature from the log-likelihood vector L.
    Uses jax.lax.dynamic_index_in_dim for a data-dependent index that is
    still JIT-safe.  Returns jnp.inf when all L values are NaN, enforces T>=1
    otherwise.
    """
    L_norm = L - jnp.nanmax(L)                            # shift so max = 0
    sorted_L = jnp.sort(L_norm)                           # NaN sorts to end in XLA
    n_valid = jnp.sum(~jnp.isnan(L)).astype(jnp.int32)
    idx = jnp.maximum(jnp.array(0, jnp.int32), n_valid - N_above - 1)
    logL_lev = jax.lax.dynamic_index_in_dim(sorted_L, idx, axis=0, keepdims=False)
    T_est = logL_lev / jnp.log(P_acc_lev)
    T_est = jnp.maximum(jnp.array(1.0), T_est)
    return jnp.where(n_valid > 0, T_est, jnp.inf)


@functools.lru_cache(maxsize=8)
def _get_postprocess_kernel(nr):
    """
    Build and cache a JIT-compiled post-processing kernel for nr samples.

    Keyed on `nr` because it determines the output shape of the uniform draws.
    Called once per data point in a Python loop — compiles for (N,) shaped
    tensors rather than (bsz, N), which avoids the minutes-long XLA fusion that
    vmap over large N would trigger.
    """

    def _postprocess_single(key, L, L_per_type_b, n_data_b, idx_jax,
                            N_above, P_acc_lev, autoT, T_base):
        """
        Full post-processing for one data point.

        Parameters
        ----------
        key            : (2,)      — PRNG key
        L              : (N,)      — combined log-likelihood
        L_per_type_b   : (Ndt, N)  — per-type log-likelihoods
        n_data_b       : (Ndt,)    — non-NaN observation count per type
        idx_jax        : (N,)      — maps sample position → original prior idx
        """
        N = L.shape[0]

        # 1. Temperature estimation
        T_auto = _logl_T_est_jax(L, N_above, P_acc_lev)
        T = jnp.where(autoT == 1, T_auto, T_base)

        # 2. Acceptance probabilities (numerically stable)
        max_L = jnp.nanmax(L)
        P_acc = jnp.exp((1.0 / T) * (L - max_L))
        P_acc = jnp.where(jnp.isnan(P_acc), 0.0, P_acc)
        p_sum = jnp.sum(P_acc)
        # Fall back to uniform when all weights collapse to zero
        p = jnp.where(p_sum > 0.0,
                      P_acc / jnp.maximum(p_sum, 1e-300),
                      jnp.ones(N) / N)

        # 3. Weighted sampling with replacement — inverse-CDF method.
        # jax.random.choice(p=p) uses gumbel-max + top_k, which generates a
        # huge XLA graph for N=1M and is very slow to compile.  The inverse-CDF
        # approach (cumsum + searchsorted) uses only two efficient GPU primitives
        # and compiles in milliseconds regardless of N.
        cdf = jnp.cumsum(p)                                    # (N,) prefix sum
        u = jax.random.uniform(key, shape=(nr,))               # (nr,) uniform draws
        i_use_raw = jnp.searchsorted(cdf, u, side='right')    # (nr,) via binary search
        i_use_raw = jnp.clip(i_use_raw, 0, N - 1)

        # 4. Evidence (log-mean-exp trick for numerical stability)
        EV = max_L + jnp.log(jnp.nanmean(jnp.exp(L - max_L)))

        # 5. Reduced chi-squared per data type
        L_accepted = L_per_type_b[:, i_use_raw]           # (Ndt, nr)
        n_data_safe = jnp.where(n_data_b > 0, n_data_b, 1.0)
        chi2_vals = jnp.nanmean(-2.0 * L_accepted, axis=1) / n_data_safe
        CHI2 = jnp.where(n_data_b > 0, chi2_vals, jnp.nan)

        # 6. Unique sample count (sort-diff avoids np.unique's dynamic shape)
        sorted_use = jnp.sort(i_use_raw)
        N_UNIQUE = (jnp.array(1, jnp.int32)
                    + jnp.sum(sorted_use[1:] != sorted_use[:-1]))

        # 7. Remap sample positions to original prior indices
        i_use = idx_jax[i_use_raw]

        return i_use, T, EV, CHI2, N_UNIQUE.astype(jnp.float32)

    return jax.jit(_postprocess_single)


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
    GPU-efficient JAX replacement for integrate_rejection_range.

    Likelihood computation and all post-processing (temperature estimation,
    weighted sampling, evidence, CHI2) run on-device.  Only the tiny final
    arrays are transferred back to the host per batch:

        i_use  (bsz × nr)  ≈ 51 KB   [was 256 MB for the likelihood matrix]
        T, EV, N_UNIQUE    ≈  1 KB
        CHI2   (bsz × Ndt) ≈  1 KB

    Full-covariance Gaussian and multinomial noise models fall back to the
    original NumPy implementations (their likelihoods are converted to JAX
    arrays once per batch before the on-device post-processing step).

    Parameters
    ----------
    D             : list of ndarray   — forward-modeled data per data type
    DATA          : dict              — observed data (same format as load_data)
    idx           : list              — prior sample indices (empty = sequential)
    N_use         : int or None       — max prior samples to evaluate
    id_use        : list              — data-type identifiers to include
    ip_range      : list              — data-point indices to invert
    nr            : int               — posterior samples per data point
    autoT         : int               — 1 = auto temperature, 0 = use T_base
    T_base        : float             — base temperature when autoT=0
    T_N_above     : int               — passed to logl_T_est (top-k for T est.)
    T_P_acc_level : float             — passed to logl_T_est (target P_acc)
    progress_callback : callable      — optional (current, total) callback
    Nbatch        : int               — data points per JAX batch (default 64)
    **kwargs      : use_N_best, showInfo, console_progress, useRandomData,
                    normalize_likelihood (bool, default False — include the
                    full Gaussian normalization constant; inert for
                    P_acc/T/EV weighting, CHI2 is de-normalized back out), …

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
    postprocess_single = _get_postprocess_kernel(nr)

    # --- Setup (mirrors integrate_rejection_range) --------------------------

    use_N_best = kwargs.get('use_N_best', 0)
    showInfo = kwargs.get('showInfo', 0)
    console_progress = kwargs.get('console_progress', True)
    disableTqdm = not console_progress if showInfo >= 0 else True
    useRandomData = kwargs.get('useRandomData', True)
    # If True, Gaussian likelihoods include the full normalization constant.
    # Inert for P_acc/T/EV weighting (constant shift); CHI2 is de-normalized
    # back out below so it stays a pure misfit statistic. See
    # integrate_rejection_range (NumPy backend) for the full rationale.
    normalize_likelihood = kwargs.get('normalize_likelihood', False)

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
    EV_post_all = np.zeros(nump) * np.nan       # not computed (kept for API compat)
    EV_post_all_mean = np.zeros(nump) * np.nan  # not computed (kept for API compat)
    CHI2_all = np.zeros((nump, Ndt)) * np.nan
    N_UNIQUE_all = np.zeros(nump) * np.nan

    # Transfer D to device once per data type (diagonal-Gaussian only)
    use_jax_diag = [
        noise_model[i] == 'gaussian'
        and DATA['Cd'][0] is None
        and DATA['d_std'][0] is not None
        for i in range(Ndt)
    ]
    D_jax = [jnp.asarray(D[i]) if use_jax_diag[i] else None for i in range(Ndt)]

    # On CPU, jnp.sort(N=1M) is ~54× slower than np.sort — the JAX on-device
    # post-processing path that is fast on GPU becomes a bottleneck on CPU.
    # Detect platform once and fall back to NumPy post-processing on CPU.
    _on_gpu = jax.local_devices()[0].platform != 'cpu'

    if _on_gpu:
        # GPU path: transfer shared scalars and idx to device once.
        idx_jax = jnp.asarray(idx) if useRandomData else jnp.arange(N, dtype=jnp.int32)
        N_above_jax   = jnp.array(T_N_above,     dtype=jnp.int32)
        P_acc_lev_jax = jnp.array(T_P_acc_level, dtype=jnp.float32)
        autoT_jax     = jnp.array(autoT,         dtype=jnp.int32)
        T_base_jax    = jnp.array(float(T_base), dtype=jnp.float32)
        rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    # --- Batch loop ---------------------------------------------------------

    for batch_start in tqdm(
        range(0, nump, Nbatch),
        disable=disableTqdm,
        desc='Rejection Sampling (JAX)',
        leave=False,
    ):
        batch_end = min(batch_start + Nbatch, nump)
        batch_js = range(batch_start, batch_end)
        ip_batch = [ip_range[j] for j in batch_js]
        bsz = len(ip_batch)

        # Build per-type log-likelihoods.
        # Diagonal-Gaussian: computed in JAX (no GPU→CPU round-trip on GPU).
        # Fallbacks (full-Cd, multinomial): computed in NumPy, then converted.
        L_per_type_list = []                               # list of (bsz, N) JAX arrays
        n_data_per_type = np.zeros((bsz, Ndt), dtype=np.float32)
        # Gaussian normalization constant used per (data point, data type),
        # 0 unless normalize_likelihood — tracked so CHI2 can be de-normalized.
        log_norm_const_b = np.zeros((bsz, Ndt))

        for i in range(Ndt):
            # active[b]=1 means data point ip_batch[b] has valid data for type i
            active = np.array([i_use_data[i][ip] for ip in ip_batch]).ravel()  # (bsz,)

            if noise_model[i] == 'gaussian':
                for b, ip in enumerate(ip_batch):
                    if active[b]:
                        n_data_per_type[b, i] = int(
                            np.sum(~np.isnan(DATA['d_obs'][i][ip]))
                        )

                if DATA['Cd'][0] is not None:
                    # Full-covariance fallback — NumPy, converted once per batch
                    L_np = np.zeros((bsz, N), dtype=np.float32)
                    for b, ip in enumerate(ip_batch):
                        if active[b]:
                            Cd = (DATA['Cd'][0][ip]
                                  if len(DATA['Cd'][0].shape) == 3
                                  else DATA['Cd'][0][:])
                            L_np[b], log_norm_const_b[b, i] = likelihood_gaussian_full(
                                D[i], DATA['d_obs'][i][ip], Cd, N_app=use_N_best,
                                normalize=normalize_likelihood, return_norm_const=True,
                            )
                    L_per_type_list.append(jnp.asarray(L_np))

                elif DATA['d_std'][0] is not None:
                    # Diagonal case: batched JAX kernel (fast on both CPU and GPU)
                    d_obs_batch = np.array([DATA['d_obs'][i][ip] for ip in ip_batch])
                    d_std_batch = np.array([DATA['d_std'][i][ip] for ip in ip_batch])
                    L_jax = likelihood_gauss_diag_batch(
                        D_jax[i],
                        jnp.asarray(d_obs_batch),
                        jnp.asarray(d_std_batch),
                    )  # (bsz, N)
                    if normalize_likelihood:
                        lnc_batch = -0.5 * np.nansum(
                            np.log(2*np.pi*d_std_batch**2), axis=1
                        ) * active
                        log_norm_const_b[:, i] = lnc_batch
                        L_jax = L_jax + jnp.asarray(lnc_batch)[:, None]
                    L_per_type_list.append(L_jax * jnp.asarray(active[:, None]))

                else:
                    L_per_type_list.append(jnp.zeros((bsz, N), dtype=jnp.float32))

            elif noise_model[i] == 'multinomial':
                # Multinomial fallback — NumPy, converted once per batch
                L_np = np.zeros((bsz, N), dtype=np.float32)
                for b, ip in enumerate(ip_batch):
                    if active[b]:
                        d_obs_ip = DATA['d_obs'][i][ip]
                        n_data_per_type[b, i] = int(np.sum(~np.isnan(d_obs_ip)))
                        L_np[b] = likelihood_multinomial(
                            D[i], d_obs_ip,
                            np.array(class_id_list[i]),
                            class_is_idx=class_is_idx,
                        )
                L_per_type_list.append(jnp.asarray(L_np))

            else:
                L_per_type_list.append(jnp.zeros((bsz, N), dtype=jnp.float32))

        # Stack to (Ndt, bsz, N) and combine
        L_per_type_stacked = jnp.stack(L_per_type_list, axis=0)  # (Ndt, bsz, N)
        L_combined = jnp.sum(L_per_type_stacked, axis=0)          # (bsz, N)

        # De-normalized per-type likelihoods, used only for CHI2 (a pure misfit
        # statistic). L_per_type_stacked (and hence L_combined, used for
        # P_acc/T/EV) stays normalized when normalize_likelihood is set.
        L_per_type_raw_stacked = L_per_type_stacked - jnp.asarray(log_norm_const_b.T)[:, :, None]

        if _on_gpu:
            # GPU path: all post-processing on-device, one point at a time.
            # Avoids the ~256 MB PCIe transfer per batch that the CPU path pays
            # for free (same-memory transfer).  A Python loop rather than vmap
            # avoids minutes-long XLA fusion for (bsz, N) shaped kernels.
            rng_key, batch_key = jax.random.split(rng_key)
            keys = jax.random.split(batch_key, bsz)               # (bsz, 2)
            n_data_jax = jnp.asarray(n_data_per_type)             # (bsz, Ndt)

            for b in range(bsz):
                i_use_b, T_b, EV_b, CHI2_b, N_UNIQUE_b = postprocess_single(
                    keys[b],
                    L_combined[b],
                    L_per_type_raw_stacked[:, b, :],
                    n_data_jax[b],
                    idx_jax,
                    N_above_jax,
                    P_acc_lev_jax,
                    autoT_jax,
                    T_base_jax,
                )
                j = batch_start + b
                i_use_all[j]    = np.asarray(i_use_b)
                T_all[j]        = float(T_b)
                EV_all[j]       = float(EV_b)
                CHI2_all[j]     = np.asarray(CHI2_b)
                N_UNIQUE_all[j] = float(N_UNIQUE_b)
        else:
            # CPU path: transfer combined likelihoods to NumPy once per batch,
            # then post-process with NumPy.  Avoids jnp.sort(N=1M) which is
            # ~54× slower than np.sort on CPU and dominates wall-clock time.
            L_combined_np    = np.asarray(L_combined)              # (bsz, N)
            L_per_type_raw_np = np.asarray(L_per_type_raw_stacked)  # (Ndt, bsz, N), de-normalized

            for b, j in enumerate(batch_js):
                L = L_combined_np[b]                               # (N,)

                if autoT == 1:
                    T = ig.logl_T_est(L, N_above=T_N_above, P_acc_lev=T_P_acc_level)
                else:
                    T = float(T_base)

                P_acc = np.exp((1.0 / T) * (L - np.nanmax(L)))
                P_acc[np.isnan(P_acc)] = 0.0
                p_sum = P_acc.sum()
                if p_sum > 0:
                    p = P_acc / p_sum
                    i_use = np.random.choice(N, nr, p=p)
                else:
                    i_use = np.random.choice(N, nr)

                CHI2_current = np.full(Ndt, np.nan)
                for i in range(Ndt):
                    if n_data_per_type[b, i] > 0:
                        L_acc = L_per_type_raw_np[i, b, i_use]
                        CHI2_current[i] = (
                            np.nanmean(-2.0 * L_acc) / n_data_per_type[b, i]
                        )

                if useRandomData:
                    i_use = idx[i_use]

                max_L = np.nanmax(L)
                EV = max_L + np.log(np.nanmean(np.exp(L - max_L)))

                i_use_all[j]    = i_use
                T_all[j]        = T
                EV_all[j]       = EV
                CHI2_all[j]     = CHI2_current
                N_UNIQUE_all[j] = len(np.unique(i_use))

        if progress_callback is not None:
            progress_callback(batch_end, nump)

    return (
        i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean,
        CHI2_all, N_UNIQUE_all, ip_range,
    )
