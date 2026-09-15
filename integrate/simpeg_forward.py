"""SimPEG 1D TDEM forward modelling as a third backend next to GA-AEM and anemone.

Public API mirrors ``forward_anemone`` / ``prior_data_anemone``: resistivity in,
positive dB/dt per unit transmitter moment [V/(A m^4)] out, one column per
used gate, LM gates first.  ``simpeg`` is imported lazily so importing
``integrate`` never requires it.

What SimPEG's ``Simulation1DLayered`` lacks for Aarhus-style systems is added
here, in the frequency domain where GA-AEM does it too:

* receiver / TiB low-pass filters -> ``_FilteredSimulation1DLayered`` multiplies
  the complex frequency response of each moment by the GEX filter chain before
  SimPEG's cosine/sine DLF (exact: the filtered response is still causal);
* gate integration -> ``gate_integration='bfield'`` samples B at the gate
  edges and takes ``(B(close) - B(open)) / (close - open)``, the exact gate
  mean of dB/dt (GA-AEM ``AreaUnderCurve``); ``'gl'`` / ``'centre'`` sample
  dB/dt instead.

See docs/superpowers/specs/2026-09-14-simpeg-forward-backend-design.md
"""
from __future__ import annotations

import multiprocessing
import os
import time

import numpy as np

from integrate.em_system import chain_response, compress_model, gex_to_em_system

_IMPORT_HINT = (
    "SimPEG backend requires 'simpeg' (>= 0.22): "
    "uv pip install -e ../simpeg  (or: pip install simpeg)"
)

# private Simulation1DLayered members this module relies on
_REQUIRED_PRIVATE = ("_project_to_data", "_compute_coefficients", "_compute_hankel_coefficients")

_DEFAULT_TX_HEIGHT_AIRBORNE = 40.0


def _require_simpeg():
    """Import simpeg lazily; raise ImportError with an install hint."""
    try:
        import simpeg
        import simpeg.electromagnetics.time_domain as tdem
    except ImportError as e:
        raise ImportError(_IMPORT_HINT) from e
    missing = [m for m in _REQUIRED_PRIVATE if not hasattr(tdem.Simulation1DLayered, m)]
    if missing:
        raise ImportError(
            f"installed simpeg {getattr(simpeg, '__version__', '?')} lacks "
            f"Simulation1DLayered.{missing[0]}; the integrate SimPEG backend needs simpeg >= 0.22")
    return simpeg, tdem


# --------------------------------------------------------------------------- #
# SimPEG subclass: frequency-domain filters + coefficient cache injection
# --------------------------------------------------------------------------- #
_SIM_CLASS = None


def _filtered_simulation_class():
    """Build (once) the Simulation1DLayered subclass; deferred so that importing
    this module never imports simpeg."""
    global _SIM_CLASS
    if _SIM_CLASS is not None:
        return _SIM_CLASS
    _, tdem = _require_simpeg()

    class _FilteredSimulation1DLayered(tdem.Simulation1DLayered):
        """``Simulation1DLayered`` with

        * ``filter_responses``: per-source list of ``[(order, fcut), ...]``
          (or ``None``); the complex frequency response of that source's
          receivers is multiplied by ``chain_response(f, filters)`` before the
          time-domain projection -- the same place GA-AEM applies its low-pass
          filters.  Valid because the product of causal responses is causal, so
          SimPEG's cosine (dB/dt, Re) and sine (B, Im/omega) transforms of the
          filtered spectrum remain exact.
        * ``coeff_cache``: ``(As, frequencies)`` from a previous
          ``get_coeff_cache()``.  These waveform x time quadrature coefficients
          are the expensive part of ``_compute_coefficients`` and do not depend
          on geometry, so they are re-used across altitudes and worker
          processes; only the (cheap) Hankel/geometry coefficients are rebuilt.
        """

        def __init__(self, *args, filter_responses=None, coeff_cache=None, **kwargs):
            super().__init__(*args, **kwargs)
            if getattr(self, "hMap", None) is not None:
                raise NotImplementedError(
                    "_FilteredSimulation1DLayered does not support hMap (its Jacobian "
                    "block takes the real part before the filter can be applied)")
            self._filter_responses = list(filter_responses or [])
            self._coeff_cache = coeff_cache
            self._G_cache = None

        # -- coefficient cache ------------------------------------------------
        def _compute_coefficients(self):
            if self._coefficients_set:
                return
            if self._coeff_cache is not None:
                As, frequencies = self._coeff_cache
                self._compute_hankel_coefficients()
                self._As = [np.asarray(A) for A in As]
                self._frequencies = np.asarray(frequencies, dtype=float)
                self._coefficients_set = True
                return
            super()._compute_coefficients()

        def get_coeff_cache(self):
            self._compute_coefficients()
            return ([A.copy() for A in self._As], self._frequencies.copy())

        # -- filters ---------------------------------------------------------
        def _source_row_blocks(self):
            """[(row0, row1), ...] of receiver-location rows per source."""
            blocks, i = [], 0
            for src in self.survey.source_list:
                n = sum(rx.locations.shape[0] for rx in src.receiver_list)
                blocks.append((i, i + n))
                i += n
            return blocks

        def _filter_gains(self):
            if self._G_cache is not None and self._G_cache[0] is self._frequencies:
                return self._G_cache[1]
            gains = []
            for filters in self._filter_responses:
                gains.append(None if not filters else chain_response(self._frequencies, filters))
            self._G_cache = (self._frequencies, gains)
            return gains

        def _project_to_data(self, v):
            if any(self._filter_responses):
                v = np.array(v, copy=True)
                gains = self._filter_gains()
                for (r0, r1), G in zip(self._source_row_blocks(), gains):
                    if G is None:
                        continue
                    if v.ndim == 3:      # (n_locs, n_omega, n_param) -- Jacobian path
                        v[r0:r1] *= G[None, :, None]
                    else:                # (n_locs, n_omega)
                        v[r0:r1] *= G[None, :]
            return super()._project_to_data(v)

    _SIM_CLASS = _FilteredSimulation1DLayered
    return _SIM_CLASS


# --------------------------------------------------------------------------- #
# survey construction
# --------------------------------------------------------------------------- #
_GATE_INTEGRATIONS = ("bfield", "gl", "centre")


def _receiver_times_and_P(moment, gate_integration="bfield", gate_quad=5):
    """Receiver sample times and the (n_gates, n_times) matrix ``P`` that turns
    the sampled SimPEG response into per-gate values.

    Returns ``(times, P, field)`` with ``field`` in {"b", "dbdt"}.
    """
    t_open = np.asarray(moment["gate_open"], dtype=float)
    t_close = np.asarray(moment["gate_close"], dtype=float)
    t_centre = np.asarray(moment["gate_centre"], dtype=float)
    ng = t_open.size
    if gate_integration == "bfield":
        # gate mean of dB/dt == (B(close) - B(open)) / width  (exact)
        times = np.concatenate([t_open, t_close])
        width = t_close - t_open
        P = np.zeros((ng, 2 * ng))
        P[np.arange(ng), np.arange(ng)] = -1.0 / width
        P[np.arange(ng), ng + np.arange(ng)] = 1.0 / width
        return times, P, "b"
    if gate_integration == "centre":
        return t_centre.copy(), np.eye(ng), "dbdt"
    if gate_integration == "gl":
        from scipy.special import roots_legendre
        K = max(int(gate_quad), 1)
        x, w = roots_legendre(K)
        # map [-1, 1] -> [open, close]; mean = sum(w/2 * f)
        times = (0.5 * (t_close - t_open)[:, None] * (x[None, :] + 1.0) + t_open[:, None]).ravel()
        P = np.zeros((ng, ng * K))
        for g in range(ng):
            P[g, g * K:(g + 1) * K] = w / 2.0
        return times, P, "dbdt"
    raise ValueError(f"unknown gate_integration {gate_integration!r}; use one of {_GATE_INTEGRATIONS}")


def _build_simulation(system, tx_height, opts, thickness, coeff_cache=None):
    """One ``LineCurrent`` per moment (loop at z=tx_height, current 1, GEX
    waveform), one receiver per moment, and the filtered simulation.

    Returns ``(sim, slices, Ps)`` where ``slices[m]`` selects moment *m*'s
    samples in ``sim.dpred`` and ``Ps[m]`` maps them onto its gates.
    """
    _, tdem = _require_simpeg()
    Sim = _filtered_simulation_class()
    tx_height = float(tx_height)
    if tx_height < 0:
        raise ValueError(f"tx_height must be >= 0 (got {tx_height})")
    loop = np.asarray(system["loop_xy"], dtype=float)
    nodes = np.column_stack([loop[:, 0], loop[:, 1], np.full(loop.shape[0], tx_height)])
    rx_x, rx_y = system["rx_xy"]
    # GEX z is positive down: rx_dz > 0 means the receiver sits *below* the Tx.
    rx_z = tx_height - float(system["rx_dz"])
    if rx_z < 0:
        raise ValueError(f"receiver height {rx_z:.2f} m is below ground for tx_height={tx_height}")

    sources, slices, Ps, filters = [], [], [], []
    i = 0
    for m in system["moments"]:
        times, P, field = _receiver_times_and_P(m, opts["gate_integration"], opts["gate_quad"])
        RxCls = (tdem.receivers.PointMagneticFluxDensity if field == "b"
                 else tdem.receivers.PointMagneticFluxTimeDerivative)
        rx = RxCls(np.array([[rx_x, rx_y, rx_z]]), times=times, orientation="z")
        wf = np.asarray(m["waveform"], dtype=float)
        waveform = tdem.sources.PiecewiseLinearWaveform(wf[:, 0], wf[:, 1])
        sources.append(tdem.sources.LineCurrent([rx], location=nodes, current=1.0, waveform=waveform))
        slices.append(slice(i, i + times.size))
        i += times.size
        Ps.append(P)
        filters.append(list(m["filters"]) if opts.get("apply_filters", True) else None)

    survey = tdem.Survey(sources)
    thickness = np.asarray(thickness, dtype=float)
    sim = Sim(
        survey=survey,
        thicknesses=thickness,
        sigma=np.full(thickness.size + 1, 1e-2),
        topo=np.r_[0.0, 0.0, 0.0],
        hankel_filter=opts["hankel_filter"],
        time_filter=opts["time_filter"],
        n_points_per_path=int(opts["n_points_per_path"]),
        filter_responses=filters,
        coeff_cache=coeff_cache,
    )
    return sim, slices, Ps


def _forward_rows(sim, M_rows, thickness, doCompress, slices, Ps, area, progress=None):
    """Evaluate the simulation for each resistivity row -> (n_rows, n_used)."""
    M_rows = np.atleast_2d(np.asarray(M_rows, dtype=float))
    thickness = np.asarray(thickness, dtype=float)
    n_used = sum(P.shape[0] for P in Ps)
    out = np.empty((M_rows.shape[0], n_used))
    for j, rho in enumerate(M_rows):
        if doCompress:
            rho_c, thk_c = compress_model(rho, thickness)
        else:
            rho_c, thk_c = rho, thickness
        sim.thicknesses = thk_c
        sim.sigma = 1.0 / rho_c
        d = sim.dpred(None)
        out[j] = np.concatenate([P @ d[sl] for sl, P in zip(slices, Ps)]) / area
        if progress is not None:
            progress(j + 1)
    return out


def _opts(kwargs_like):
    return {
        "gate_integration": kwargs_like.get("gate_integration", "bfield"),
        "gate_quad": int(kwargs_like.get("gate_quad", 5)),
        "hankel_filter": kwargs_like.get("hankel_filter", "key_101_2009"),
        "time_filter": kwargs_like.get("time_filter", "key_81_2009"),
        "n_points_per_path": int(kwargs_like.get("n_points_per_path", 2)),
        "apply_filters": bool(kwargs_like.get("apply_filters", True)),
    }


# --------------------------------------------------------------------------- #
# chunk worker (module level: picklable for multiprocessing)
# --------------------------------------------------------------------------- #
def _forward_simpeg_chunk(M_chunk, txh_chunk, thickness, system, opts, doCompress,
                          area, coeff_cache):
    """Forward one chunk; ``txh_chunk`` is already binned (few unique heights)."""
    M_chunk = np.atleast_2d(np.asarray(M_chunk, dtype=float))
    txh_chunk = np.asarray(txh_chunk, dtype=float).ravel()
    n_used = None
    out = None
    for h in np.unique(txh_chunk):
        rows = np.where(txh_chunk == h)[0]
        sim, slices, Ps = _build_simulation(system, h, opts, thickness, coeff_cache=coeff_cache)
        d = _forward_rows(sim, M_chunk[rows], thickness, doCompress, slices, Ps, area)
        if out is None:
            n_used = d.shape[1]
            out = np.empty((M_chunk.shape[0], n_used))
        out[rows] = d
    return out


# --------------------------------------------------------------------------- #
# public forward
# --------------------------------------------------------------------------- #
def forward_simpeg(M=np.array(()), thickness=np.array(()), file_gex=None, GEX=None,
                   tx_height=np.array(()), altitude_bin_width=1.0, is_log=False,
                   doCompress=True, gate_integration="bfield", gate_quad=5,
                   hankel_filter="key_101_2009", time_filter="key_81_2009",
                   n_points_per_path=2, parallel=True, Ncpu=0,
                   showInfo=0, showtime=False, progress_callback=None, **kwargs):
    """
    Forward TDEM response of layered models with SimPEG.

    Parameters
    ----------
    M : ndarray (nd, nl) or (nl,)
        Layer resistivities [ohm m].
    thickness : ndarray (nl-1,)
        Layer thicknesses [m] (last layer = half-space).
    file_gex : str, optional
        Aarhus ``.gex`` system file.  Alternatively pass ``GEX`` (dict from
        ``read_gex_workbench``) or a system dict from ``gex_to_em_system``.
    tx_height : float or ndarray (nd,), optional
        Transmitter height above ground [m].  Default: ``-TxCoilPosition1[2]``
        for ground systems (tTEM), else 40 m.
    altitude_bin_width : float
        Varying ``tx_height`` is rounded to bins of this width (m) so that one
        SimPEG geometry serves many soundings; ``<= 0`` uses exact heights.
    is_log : bool
        Return ``log10`` of the data.
    doCompress : bool
        Merge adjacent layers with identical resistivity before each forward
        (exact; fewer layers = faster).
    gate_integration : {'bfield', 'gl', 'centre'}
        How a gate value is obtained: exact gate mean of dB/dt from B at the
        gate edges (default), Gauss-Legendre mean of dB/dt with ``gate_quad``
        points, or dB/dt at the gate centre.
    hankel_filter, time_filter : str
        SimPEG/libdlf filter names (``'key_101_2009'`` / ``'key_81_2009'``).
    n_points_per_path : int
        Gauss-Legendre points per loop segment (SimPEG ``LineCurrent``).
    parallel : bool
        Use a multiprocessing pool over soundings (``Ncpu`` = 0 -> all cores).
    progress_callback : callable, optional
        ``progress_callback(current, total, info_dict)``.
    **kwargs
        Ignored (accepts GA-AEM-only options such as ``Nhank``/``Nfreq``).

    Returns
    -------
    ndarray (nd, n_used) -- or (n_used,) for a 1-D ``M``
        dB/dt per unit transmitter moment [V/(A m^4)], positive, LM gates then HM.
    """
    from integrate.integrate import _report_progress

    _require_simpeg()
    t0 = time.time()

    if isinstance(GEX, dict) and "moments" in GEX:
        system = GEX
    else:
        system = gex_to_em_system(GEX if GEX is not None else file_gex, showInfo=showInfo)

    M = np.asarray(M, dtype=float)
    one_d = M.ndim == 1
    M = np.atleast_2d(M)
    nd, nl = M.shape
    thickness = np.asarray(thickness, dtype=float).ravel()
    if thickness.size != nl - 1:
        raise ValueError(f"thickness has {thickness.size} entries, expected nl-1 = {nl - 1}")

    tx_height = np.asarray(tx_height, dtype=float).ravel()
    if tx_height.size == 0:
        h0 = system["tx_z"] if system["has_tx_coil_position"] else _DEFAULT_TX_HEIGHT_AIRBORNE
        tx_height = np.full(nd, h0)
    elif tx_height.size == 1:
        tx_height = np.full(nd, tx_height[0])
    elif tx_height.size != nd:
        raise ValueError(f"tx_height length {tx_height.size} != number of soundings {nd}")
    varying = tx_height.size > 1 and not np.allclose(tx_height, tx_height[0])
    if varying and altitude_bin_width and altitude_bin_width > 0:
        tx_binned = np.round(tx_height / altitude_bin_width) * altitude_bin_width
        # a bin must keep the receiver (tx_height - rx_dz) at or above ground
        tx_binned = np.maximum(tx_binned, max(0.0, system["rx_dz"]))
    else:
        tx_binned = tx_height.copy()
    n_bins = np.unique(tx_binned).size

    opts = _opts(dict(gate_integration=gate_integration, gate_quad=gate_quad,
                      hankel_filter=hankel_filter, time_filter=time_filter,
                      n_points_per_path=n_points_per_path,
                      apply_filters=kwargs.get("apply_filters", True)))
    area = system["tx_area_poly"]

    # Waveform/time coefficients once, in the parent (geometry-independent).
    sim0, slices, Ps = _build_simulation(system, tx_binned[0], opts, thickness)
    coeff_cache = sim0.get_coeff_cache()
    n_used = sum(P.shape[0] for P in Ps)
    if showInfo:
        print(f"forward_simpeg: nd={nd}, nl={nl}, n_used={n_used}, altitude bins={n_bins}, "
              f"gate_integration={gate_integration}, setup {time.time() - t0:.2f}s")

    Ncpu = int(Ncpu or kwargs.get("Nproc", 0) or 0)
    if Ncpu <= 0:
        Ncpu = os.cpu_count() or 1
    in_main = multiprocessing.current_process().name == "MainProcess"
    use_pool = bool(parallel) and in_main and nd >= 2 * Ncpu and Ncpu > 1

    _report_progress(progress_callback, 0, nd, "computing", "SimPEG forward")
    D = np.empty((nd, n_used))
    if not use_pool:
        # sequential: reuse sim0 for its bin, build others from the cache
        done = [0]

        def _prog(k):
            done[0] += 1
            if progress_callback is not None and (done[0] % 50 == 0 or done[0] == nd):
                _report_progress(progress_callback, done[0], nd, "computing", "SimPEG forward")

        for h in np.unique(tx_binned):
            rows = np.where(tx_binned == h)[0]
            if h == tx_binned[0]:
                sim = sim0
            else:
                sim, slices, Ps = _build_simulation(system, h, opts, thickness, coeff_cache=coeff_cache)
            D[rows] = _forward_rows(sim, M[rows], thickness, doCompress, slices, Ps, area,
                                    progress=_prog)
    else:
        n_chunks = Ncpu * 4 if progress_callback is not None else Ncpu
        n_chunks = min(n_chunks, nd)
        # sort by height so chunks contain few distinct geometries
        order = np.argsort(tx_binned, kind="stable")
        idx_chunks = np.array_split(order, n_chunks)
        ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        if showInfo:
            print(f"forward_simpeg: {Ncpu} processes, {n_chunks} chunks")
        with ctx.Pool(processes=Ncpu) as pool:
            jobs = [
                pool.apply_async(_forward_simpeg_chunk,
                                 (M[idx], tx_binned[idx], thickness, system, opts, doCompress,
                                  area, coeff_cache))
                for idx in idx_chunks
            ]
            done = 0
            for idx, job in zip(idx_chunks, jobs):
                D[idx] = job.get()
                done += idx.size
                _report_progress(progress_callback, done, nd, "computing", "SimPEG forward")

    if is_log:
        D = np.log10(D)
    if showtime or showInfo:
        dt = time.time() - t0
        print(f"forward_simpeg: Time={dt:6.1f}s/{nd} soundings, {1000 * dt / max(nd, 1):.1f}ms/sounding")
    return D[0] if one_d else D


# --------------------------------------------------------------------------- #
# prior data generator
# --------------------------------------------------------------------------- #
def prior_data_simpeg(f_prior_h5, file_gex=None, N=0, doMakePriorCopy=True, im=1, id=1,
                      im_height=0, is_log=False, altitude_bin_width=1.0, doCompress=True,
                      gate_integration="bfield", gate_quad=5,
                      hankel_filter="key_101_2009", time_filter="key_81_2009",
                      n_points_per_path=2, parallel=True, Ncpu=0,
                      force_replace=False, f_prior_data_h5="", randomize=True, **kwargs):
    """
    Generate prior data ``/D{id}`` with the SimPEG forward backend.

    Same contract as ``prior_data_gaaem`` / ``prior_data_anemone``: reads
    resistivity ``/M{im}`` (and ``/M{im_height}`` as transmitter height when
    ``im_height > 0``) from ``f_prior_h5``, optionally copies the file first
    (``doMakePriorCopy``; auto-name ``<stem>_<gex>[_N<N>]_simpeg.h5``), writes
    ``/D{id}`` = dB/dt per unit transmitter moment, and returns the file path.

    See :func:`forward_simpeg` for the modelling options.  ``showInfo`` and
    ``progress_callback`` may be passed as keyword arguments.
    """
    import h5py
    import integrate as ig
    from integrate.integrate import _report_progress

    if multiprocessing.current_process().name != "MainProcess":
        return None

    simpeg, _ = _require_simpeg()
    showInfo = kwargs.get("showInfo", 0)
    progress_callback = kwargs.pop("progress_callback", None)

    with h5py.File(f_prior_h5, "r") as f:
        N_in = f["M1"].shape[0]
    if N == 0 or N > N_in:
        N = N_in

    if file_gex is not None and not os.path.isfile(file_gex):
        print("ERROR: file_gex=%s does not exist" % file_gex)

    if doMakePriorCopy:
        if not f_prior_data_h5:
            base = (os.path.splitext(os.path.basename(file_gex))[0]
                    if file_gex else "SIMPEG")
            stem = os.path.splitext(f_prior_h5)[0]
            f_prior_data_h5 = ("%s_%s_N%d_simpeg.h5" % (stem, base, N)
                               if N < N_in else
                               "%s_%s_simpeg.h5" % (stem, base))
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5, N, randomize=randomize, showInfo=showInfo)
    else:
        f_prior_data_h5 = f_prior_h5

    Mname, Dname = "/M%d" % im, "/D%d" % id
    Mheight = "/M%d" % im_height

    with h5py.File(f_prior_data_h5, "r") as f:
        zattr = f[Mname].attrs["x" if "x" in f[Mname].attrs else "z"]
        thickness = np.diff(np.asarray(zattr, dtype=float))
        M = np.asarray(f[Mname][:], dtype=float)          # resistivity
        tx_height = (np.asarray(f[Mheight][:], dtype=float).ravel()
                     if im_height > 0 else np.array(()))

    t1 = time.time()
    D = forward_simpeg(M=M, thickness=thickness, file_gex=file_gex, tx_height=tx_height,
                       altitude_bin_width=altitude_bin_width, is_log=is_log,
                       doCompress=doCompress, gate_integration=gate_integration,
                       gate_quad=gate_quad, hankel_filter=hankel_filter,
                       time_filter=time_filter, n_points_per_path=n_points_per_path,
                       parallel=parallel, Ncpu=Ncpu, showInfo=showInfo,
                       progress_callback=progress_callback)
    if showInfo > -1:
        dt = time.time() - t1
        n_sound = M.shape[0]
        print("prior_data_simpeg: Time=%5.1fs/%d soundings. %4.1fms/sounding, %3.1fit/s"
              % (dt, n_sound, 1000 * dt / max(n_sound, 1), n_sound / max(dt, 1e-12)))

    _report_progress(progress_callback, N, N, "saving",
                     "Saving forward data to %s" % f_prior_data_h5)

    with h5py.File(f_prior_data_h5, "a") as f:
        if Dname in f:
            if force_replace:
                del f[Dname]
            else:
                print("Key '%s' already exists in %s. Use force_replace=True."
                      % (Dname, f_prior_data_h5))
                return f_prior_data_h5
        f[Dname] = D
        a = f[Dname].attrs
        a["method"] = "simpeg"
        a["type"] = "TDEM"
        a["im"] = im
        a["id"] = id
        a["is_log"] = bool(is_log)
        a["altitude_bin_width"] = float(altitude_bin_width)
        a["doCompress"] = bool(doCompress)
        a["gate_integration"] = str(gate_integration)
        a["gate_quad"] = int(gate_quad)
        a["hankel_filter"] = str(hankel_filter)
        a["time_filter"] = str(time_filter)
        a["n_points_per_path"] = int(n_points_per_path)
        a["simpeg_version"] = str(getattr(simpeg, "__version__", "unknown"))
        if file_gex:
            a["file_gex"] = os.path.basename(file_gex)

    ig.integrate_update_prior_attributes(f_prior_data_h5)
    _report_progress(progress_callback, N, N, "completed",
                     "Forward data saved to %s" % f_prior_data_h5)
    return f_prior_data_h5
