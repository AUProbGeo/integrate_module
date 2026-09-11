"""anemone (PyTorch) 1D TDEM forward modelling as an alternative to GA-AEM.

Public API mirrors ``forward_gaaem`` / ``prior_data_gaaem`` in
``integrate.integrate``.  ``anemone`` and ``torch`` are imported lazily so
importing ``integrate`` never requires them.

See docs/superpowers/specs/2026-09-10-anemone-forward-backend-design.md
"""
from __future__ import annotations

import os

import numpy as np

_IMPORT_HINT = (
    "anemone backend requires 'anemone' and 'torch': "
    "pip install torch && pip install -e /home/tmeha/PROGRAMMING/anemone"
)


def _require_anemone():
    """Import and return ``(anemone, torch)`` or raise a clear ImportError."""
    try:
        import torch  # noqa: F401
        import anemone  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_IMPORT_HINT) from exc
    return anemone, torch


def _load_gex(gex):
    """Return a libaarhusxyz GEX object, parsing a path or wrapping a dict."""
    if isinstance(gex, str):
        try:
            from libaarhusxyz import GEX
            return GEX(gex)
        except Exception:
            import integrate as ig
            return _DictGex(ig.read_gex_workbench(gex))
    if isinstance(gex, dict):
        return _DictGex(gex)
    return gex  # already a libaarhusxyz GEX


class _DictGex:
    """Minimal libaarhusxyz.GEX-shaped accessor over an INTEGRATE GEX dict."""

    def __init__(self, d):
        self.gex_dict = d
        self._nch = 2 if any(k.startswith("GateTimeHM") or k == "Channel2"
                             for k in list(d) + list(d.get("General", {}))) else 1

    @property
    def number_channels(self):
        return self._nch

    def _gate_array(self, ch):
        G = self.gex_dict["General"]
        for key in (f"GateArray{'LM' if ch == 1 else 'HM'}", "GateArrayLM",
                    "GateArray"):
            if key in G:
                return np.atleast_2d(np.asarray(G[key], dtype=float))
        raise KeyError("no gate array in GEX dict")

    def gate_times(self, ch):
        return self._gate_array(ch)

    def _chan(self, ch):
        return self.gex_dict.get(f"Channel{ch}", {})

    def remove_initial_gates(self, ch):
        return float(np.atleast_1d(self._chan(ch).get("RemoveInitialGates", 0))[0])

    def no_gates(self, ch):
        return float(np.atleast_1d(self._chan(ch).get("NoGates",
                    self._gate_array(ch).shape[0]))[0])


def _butter_rows(arr):
    """Normalise an RxCoilLPFilter value to a list of (order, fcut) rows."""
    a = np.atleast_2d(np.asarray(arr, dtype=float))
    if a.shape == (1, 2):
        return [(a[0, 0], a[0, 1])]
    return [(row[0], row[1]) for row in a]


def _polygon_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _gex_signature(g):
    import hashlib
    d = getattr(g, "gex_dict", {})
    return hashlib.md5(repr(sorted(str(d).encode())[:0] or str(d)).encode()).hexdigest()


_FORWARD_CACHE = {}


def _clear_forward_cache():
    _FORWARD_CACHE.clear()


def _remake_loop(loop, tx_z, torch):
    from anemone.system import Loop
    if tx_z is None:
        return loop
    z = torch.full_like(loop.x, float(tx_z))
    return Loop(loop.x.clone(), loop.y.clone(), z)


def _filter_sig(fc):
    """Return signature (fcut, order) tuples for a FilterChain."""
    return [(round(float(f.fcut), 3), int(f.order)) for f in fc]


def _build_forward(system, tx_z, device):
    _, torch = _require_anemone()
    from anemone.forwards import Forward
    from anemone.system import Receiver

    n_moments = len(system["moments"])
    if n_moments not in (1, 2):
        raise NotImplementedError(
            "anemone backend: only 1 or 2 moments supported")

    tx_z_key = "src" if tx_z is None else round(float(tx_z), 3)
    key = (system["gex_signature"], tx_z_key, str(device))
    if key in _FORWARD_CACHE:
        return _FORWARD_CACHE[key]

    dev = torch.device(device)

    # Hoist and build shared source, receiver, filterfunc from first moment
    m0 = system["moments"][0]
    shared_src = _remake_loop(m0["source"], tx_z, torch)
    # The Rx coil position in the GEX is an OFFSET from the Tx frame, so the
    # receiver height tracks the loop height: receiver_z = loop_z + dz.
    # anemone uses sfield_dz = |srcz| + |rz| and pfield_dz = |srcz - rz|, so
    # both must be positive heights above ground (parity with GA-AEM's
    # tx_height / txrx_dz).
    loop_z = float(shared_src.z[0])
    dx, dy, _dz = system.get("rx_offset", (0.0, 0.0, 0.0))
    receiver_z = loop_z + float(system.get("rx_offset_z_rel_tx", 0.0))
    shared_rcv = Receiver(x=float(dx), y=float(dy), z=float(receiver_z))
    shared_filt = m0["filterfunc"]
    shared_filt_sig = _filter_sig(shared_filt)

    parts = []
    fwr = None
    for i, m in enumerate(system["moments"]):
        src = shared_src
        rcv = shared_rcv

        # Per-moment filter guard: verify identical receiver filters
        if i > 0:
            if _filter_sig(m["filterfunc"]) != shared_filt_sig:
                raise NotImplementedError(
                    "anemone backend: per-moment receiver filters differ; a combined "
                    "dual-moment Forward requires identical filters across moments")
        filt = shared_filt

        # Each moment uses its own waveform (LM vs HM transmitter waveforms differ)
        wf = m["waveform"]
        times = torch.as_tensor(m["gate_times"], dtype=torch.float64).to(dev)
        part = Forward(src, rcv, times, wf, filt, tolerance=1e-6)
        parts.append(part)
        fwr = part if fwr is None else (fwr + part)

    if n_moments == 2:
        slices = [fwr.t1slc, fwr.t2slc]
    else:
        slices = [slice(None)]

    # Expose the per-moment Forward objects for introspection/tests (the
    # combined Forward keeps only the first moment's waveform).
    fwr.moment_parts = parts

    _FORWARD_CACHE[key] = (fwr, slices)
    return _FORWARD_CACHE[key]


def gex_to_anemone_system(gex, showInfo=0):
    anemone, torch = _require_anemone()
    from anemone.system import (Loop, Receiver, Waveform, FilterChain,
                                ButterworthFilter)

    g = _load_gex(gex)
    G = g.gex_dict["General"]
    n_moments = int(g.number_channels)

    has_txpos = any(k.startswith("TxCoilPosition") for k in G)
    if has_txpos:
        txkey = "TxCoilPosition1" if "TxCoilPosition1" in G else "TxCoilPosition"
        tx_pos_z = float(np.atleast_1d(np.asarray(G[txkey], dtype=float))[2])
        tx_z = abs(tx_pos_z)
    else:
        tx_pos_z = 0.0
        tx_z = 0.0

    rxkey = "RxCoilPosition1" if "RxCoilPosition1" in G else "RxCoilPosition"
    # Signed Rx coil position; it is an OFFSET from the transmitter frame, not
    # an absolute height (parity with forward_gaaem's txrx_dx/dy/dz).
    rx_offset = np.atleast_1d(np.asarray(G[rxkey], dtype=float))
    if has_txpos:  # ground system (tTEM): offset relative to the Tx coil
        rx_offset_z_rel_tx = float(rx_offset[2]) - tx_pos_z
    else:          # airborne system (SkyTEM): offset relative to the Tx frame
        rx_offset_z_rel_tx = float(rx_offset[2])

    tx_pts = G.get("TxLoopPoint")
    if tx_pts is None:  # dict form: TxLoopPoint1..N
        keys = sorted((k for k in G if k.startswith("TxLoopPoint")),
                      key=lambda k: int(k.replace("TxLoopPoint", "")))
        tx_pts = np.array([np.atleast_1d(G[k])[:2] for k in keys], dtype=float)
    tx_pts = np.asarray(tx_pts, dtype=float)
    xs = np.append(tx_pts[:, 0], tx_pts[0, 0])
    ys = np.append(tx_pts[:, 1], tx_pts[0, 1])
    tx_area_poly = _polygon_area(tx_pts[:, 0], tx_pts[:, 1])
    tx_area_gex = float(G.get("TxLoopArea", tx_area_poly))
    if showInfo and abs(tx_area_poly - tx_area_gex) / max(tx_area_gex, 1e-9) > 0.05:
        print(f"gex_to_anemone_system: polygon area {tx_area_poly:.3g} vs "
              f"TxLoopArea {tx_area_gex:.3g}")

    rx_rows = _butter_rows(G["RxCoilLPFilter"]) if "RxCoilLPFilter" in G else []

    moments = []
    for ch in range(1, n_moments + 1):
        chan = g.gex_dict.get(f"Channel{ch}", {})
        name = str(chan.get("TransmitterMoment",
                            "LM" if ch == 1 else "HM")).strip() or None

        zs = np.full_like(xs, tx_z)
        source = Loop(torch.tensor(xs), torch.tensor(ys), torch.tensor(zs))
        # Placeholder receiver; the real one is rebuilt per tx_z in
        # _build_forward() at loop_z + rx_offset_z_rel_tx.
        receiver = Receiver(x=float(rx_offset[0]), y=float(rx_offset[1]),
                            z=float(tx_z + rx_offset_z_rel_tx))

        # Waveform/turns are keyed off the CHANNEL INDEX, never the free-text
        # TransmitterMoment label (which is only used as the moment's name).
        mom = "LM" if ch == 1 else "HM"
        wf_key = "WaveformLM" if mom == "LM" else "WaveformHM"
        wf = G.get(wf_key)
        if wf is None:
            wf = G.get(f"{wf_key}Point")  # Try WaveformLMPoint or WaveformHMPoint
        if wf is None:
            keys = sorted((k for k in G if k.startswith(f"{wf_key}Point")),
                          key=lambda k: int(k.replace(f"{wf_key}Point", "") or "0"))
            wf = np.array([np.atleast_1d(G[k])[:2] for k in keys], dtype=float)
        wf = np.asarray(wf, dtype=float)
        waveform = Waveform(torch.tensor(wf[:, 0]), torch.tensor(wf[:, 1]))

        filt = []
        tib = chan.get("TiBLowPassFilter")
        if tib is not None:
            tib = np.atleast_1d(np.asarray(tib, dtype=float))
            filt.append(ButterworthFilter(float(tib[1]), int(round(tib[0]))))
        for order, fcut in rx_rows:
            filt.append(ButterworthFilter(float(fcut), int(round(order))))
        filterfunc = FilterChain(filt)

        i0 = int(g.remove_initial_gates(ch))
        i1 = int(g.no_gates(ch))
        gate_times = np.asarray(g.gate_times(ch), dtype=float)[i0:i1, 0]

        turns_key = "NumberOfTurnsLM" if mom == "LM" else "NumberOfTurnsHM"
        moments.append({
            "name": name,
            "source": source,
            "receiver": receiver,
            "waveform": waveform,
            "filterfunc": filterfunc,
            "gate_times": gate_times,
            "n_turns": float(np.atleast_1d(G.get(turns_key, 1))[0]),
            "tx_current": float(np.atleast_1d(
                chan.get("TxApproximateCurrent", 1.0))[0]),
            "tx_area": tx_area_poly,
            "front_gate_delay": float(np.atleast_1d(
                G.get("FrontGateDelay", 0.0))[0]),
        })

    return {
        "n_moments": n_moments,
        "has_tx_coil_position": bool(has_txpos),
        "rx_offset": (float(rx_offset[0]), float(rx_offset[1]),
                      float(rx_offset[2])),
        "rx_offset_z_rel_tx": float(rx_offset_z_rel_tx),
        "moments": moments,
        "gex_signature": _gex_signature(g),
        "gex": g,
    }


def _used_gate_times(system):
    times = [m["gate_times"] for m in system["moments"]]
    names = [m["name"] or f"CH{i + 1}" for i, m in enumerate(system["moments"])]
    return times, names


def _moment_scale(system, calibration_factor):
    """Per-moment factor turning anemone's raw output into ``dB/dt [V/(A m^4)]``.

    anemone returns dBz/dt for the real loop polygon carrying 1 A x 1 turn,
    i.e. for transmitter moment ``A_tx``.  The Workbench/AarhusInv data unit
    ``V/(A m^4)`` is dB/dt **per unit transmitter moment**, so the conversion
    is simply ``raw / A_tx`` -- no current or turns involved (they cancel).
    Verified on Daugaard tTEM and Marocco SkyTEM to 1-4 % (see repo
    ``ANEMONE_VS_GAAEM_VS_AI.md``).  ``k`` (default 1) is an optional extra
    multiplier per moment; ``_fit_calibration`` estimates it as a *check*.
    """
    cf = calibration_factor or {}
    out = []
    for i, m in enumerate(system["moments"]):
        name = m["name"] or f"CH{i + 1}"
        if cf:
            if name in cf:
                k = float(cf[name])
            elif i in cf:
                k = float(cf[i])
            else:
                raise ValueError(
                    f"calibration_factor is missing moment '{name}' "
                    f"(have keys {list(cf)})")
        else:
            k = 1.0
        out.append(k / m["tx_area"])
    return out  # sign is +1 in the evaluator (parity with ga-aem's -fm.SZ)


def _compress_batch(M, thickness):
    """Merge adjacent layers with identical resistivity, per sounding.

    Mirrors ``forward_gaaem``'s ``doCompress``: a run of adjacent layers with the
    same resistivity is replaced by one layer of the summed thickness.  Because
    every sounding compresses to a different layer count / thickness vector, the
    result is padded to a common ``K`` with zero-thickness trailing layers
    (physically transparent in anemone -- verified exact), so the whole batch
    still goes through one ``Forward`` call with per-model thicknesses.

    Parameters
    ----------
    M : ndarray (nd, nl)      resistivity
    thickness : ndarray (nl-1,)   shared fine-grid thicknesses

    Returns
    -------
    M_c : ndarray (nd, K)         compressed + padded resistivity
    T_c : ndarray (nd, K-1)       per-sounding compressed thicknesses (0-padded)
    """
    M = np.asarray(M, dtype=float)
    nd, nl = M.shape
    thickness = np.asarray(thickness, dtype=float)
    cz = np.concatenate([[0.0], np.cumsum(thickness)])          # interface depths
    comp = []
    for row in M:
        edges = np.concatenate([[0], np.where(np.diff(row) != 0)[0] + 1])
        comp.append((row[edges], np.diff(cz[edges])))          # (k,), (k-1,)
    # anemone's RTE recursion needs >= 2 layers; never exceed the input count
    K = min(max(max(len(r) for r, _ in comp), 2), nl)
    M_c = np.empty((nd, K), dtype=float)
    T_c = np.zeros((nd, K - 1), dtype=float)
    for j, (rho_c, thk_c) in enumerate(comp):
        k = len(rho_c)
        M_c[j, :k] = rho_c
        M_c[j, k:] = rho_c[-1]                                 # pad = half-space
        T_c[j, :k - 1] = thk_c
    return M_c, T_c


def _raw_by_moment(system, M, thickness, tx_height, device):
    """Uncalibrated raw dB/dt per moment, shape (nd, n_used_m).

    anemone raw and ``forward_gaaem`` are both positive -> no sign flip
    (Controller Ruling 4).
    """
    anemone, torch = _require_anemone()
    txh = np.asarray(tx_height, dtype=float).ravel()
    tx_z = None
    if txh.size >= 1:
        # The calibration reference is the FIRST sounding (M[:1]), so fit at
        # that sounding's altitude, not the survey median.
        tx_z = float(txh[0])
    elif not system["has_tx_coil_position"]:
        tx_z = 40.0
    fwr, slices = _build_forward(system, tx_z, device)
    M_t = torch.as_tensor(np.atleast_2d(M), dtype=torch.float64).to(
        torch.device(device))
    thk_t = torch.as_tensor(np.asarray(thickness, dtype=float),
                            dtype=torch.float64).to(torch.device(device))
    with torch.no_grad():
        raw = fwr(M_t.T, thk_t).detach().cpu().numpy()
    raw = np.atleast_2d(raw)  # anemone squeezes model axis for n_models==1
    return [(raw[:, s]) for s in slices]


def _loglog_resample(x_src, y_src, x_dst):
    good = np.isfinite(y_src) & (y_src != 0)
    if good.sum() < 2:
        return y_src
    if x_src.shape == x_dst.shape and np.allclose(x_src, x_dst):
        return y_src
    sign = np.sign(np.nanmedian(y_src[good]))
    return sign * 10 ** np.interp(np.log10(x_dst), np.log10(x_src[good]),
                                  np.log10(np.abs(y_src[good])))


def _fit_calibration(system, M, thickness, tx_height, device, reference, tol,
                     showInfo=0):
    raw_by_m = _raw_by_moment(system, M, thickness, tx_height, device)
    times, names = _used_gate_times(system)
    k_out, resid_out = {}, {}
    for i, name in enumerate(names):
        if name not in reference:
            raise ValueError(f"calibration_reference missing moment '{name}'")
        ref = np.asarray(reference[name], dtype=float)
        ref_t = np.asarray(reference.get(name + "_times", times[i]), dtype=float)
        raw_m = np.asarray(raw_by_m[i][0], dtype=float)  # first model row
        raw_on_ref = _loglog_resample(np.asarray(times[i], dtype=float), raw_m,
                                      ref_t)
        det = 1.0 / system["moments"][i]["tx_area"]   # per-unit-moment convention
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(ref) / np.abs(det * raw_on_ref)
        good = np.isfinite(r) & (r > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = float(np.exp(np.median(np.log(r[good])))) if good.any() else np.nan
            model = k * det * raw_on_ref
            rel = np.abs(np.abs(model) - np.abs(ref)) / np.abs(ref)
        rel_fin = rel[np.isfinite(rel)]
        resid = float(np.median(rel_fin)) if rel_fin.size else np.nan
        if good.sum() == 0 or not np.isfinite(k) or not np.isfinite(resid):
            forward_anemone.last_calibration = {
                "mode": "failed", "k": dict(k_out), "residual": dict(resid_out)}
            raise RuntimeError(
                f"anemone calibration for {name}: could not fit "
                f"(no finite gates / nan)")
        if resid > tol:
            forward_anemone.last_calibration = {
                "mode": "failed", "k": dict(k_out), "residual": dict(resid_out)}
            raise RuntimeError(
                f"anemone calibration for {name}: residual {resid:.3f} "
                f"> tol {tol:.3f}")
        if showInfo > 0:
            print(f"anemone calibration {name}: k={k:.4g} residual={resid:.3f}")
        k_out[name], resid_out[name] = k, resid
    return k_out, resid_out


def forward_anemone(M=np.array(()), thickness=np.array(()), file_gex=None,
                    GEX=None, tx_height=np.array(()), altitude_bin_width=1.0,
                    is_log=False, device="cpu", calibration="auto",
                    calibration_reference=None, calibration_factor=None,
                    calibration_tol=0.05, doCompress=True, showtime=False,
                    showInfo=0, progress_callback=None, **kwargs):
    anemone, torch = _require_anemone()
    import time

    M = np.asarray(M, dtype=float)
    thickness = np.asarray(thickness, dtype=float)
    one_d = M.ndim == 1
    if one_d:
        M = M[None, :]
    nd, nl = M.shape
    if thickness.shape[0] != nl - 1:
        raise ValueError(
            "thickness array (nt=%d) does not match number of layers minus 1 "
            "(nl=%d)" % (thickness.shape[0], nl))

    system = gex_to_anemone_system(GEX if GEX is not None else file_gex,
                                   showInfo=showInfo)

    tx_height = np.asarray(tx_height, dtype=float).ravel()
    if tx_height.size > 1 and tx_height.size != nd:
        raise ValueError(
            f"tx_height length {tx_height.size} != number of soundings {nd}")
    varying = tx_height.size > 1 and not np.allclose(tx_height, tx_height[0])

    # Output is dB/dt per unit transmitter moment [V/(A m^4)] straight from the
    # GEX (mode "gex", k=1).  An explicit calibration_factor overrides k; a
    # calibration_reference fits k as a check (should come out ~1).
    _, names = _used_gate_times(system)
    if calibration_factor:
        k_by_moment = dict(calibration_factor)
        forward_anemone.last_calibration = {
            "mode": "factor", "k": dict(calibration_factor), "residual": {}}
    elif calibration_reference is not None:  # auto / fitted + reference -> fit
        k_by_moment, resid = _fit_calibration(
            system, M[:1], thickness, tx_height, device,
            calibration_reference, calibration_tol, showInfo)
        forward_anemone.last_calibration = {
            "mode": "fitted", "k": k_by_moment, "residual": resid}
    elif calibration == "fitted":  # explicit fit requested but nothing to fit to
        raise ValueError(
            "anemone calibration='fitted' needs calibration_reference (dict of "
            "per-moment reference dB/dt) or calibration_factor")
    else:  # "auto" / "gex": per-unit-moment from the GEX, nothing to fit
        k_by_moment = None
        forward_anemone.last_calibration = {
            "mode": "gex", "k": {n: 1.0 for n in names}, "residual": {}}

    scale = _moment_scale(system, k_by_moment)

    # Compress adjacent identical layers (exact; mirrors forward_gaaem doCompress).
    # M_c (nd, K) resistivity, T_c (nd, K-1) per-sounding thicknesses.
    if doCompress and nl > 2:
        M_c, T_c = _compress_batch(M, thickness)
        if showInfo > 0:
            print("forward_anemone: compressed %d -> %d layers (batch max)"
                  % (nl, M_c.shape[1]))
    else:
        M_c = M
        T_c = np.tile(np.asarray(thickness, dtype=float), (nd, 1))

    t0 = time.time()
    if varying:
        D = _forward_varying_height(system, M_c, T_c, tx_height,
                                    altitude_bin_width, scale, device,
                                    progress_callback, showInfo)  # Task 6
    else:
        tx_z = None
        if tx_height.size >= 1:
            tx_z = float(tx_height[0])
        elif not system["has_tx_coil_position"]:
            tx_z = 40.0
        fwr, slices = _build_forward(system, tx_z, device)
        dev = torch.device(device)
        M_t = torch.as_tensor(M_c, dtype=torch.float64).to(dev)
        T_t = torch.as_tensor(T_c, dtype=torch.float64).to(dev)
        with torch.no_grad():
            raw = fwr(M_t.T, T_t.T)
        raw = np.asarray(raw.detach().cpu().numpy(), dtype=float)
        raw = np.atleast_2d(raw)
        cols = []
        for s, sc in zip(slices, scale):
            cols.append(sc * raw[:, s])
        D = np.concatenate(cols, axis=1)

    if showtime:
        print("forward_anemone: %.1f ms/model (%d models)"
              % (1000 * (time.time() - t0) / nd, nd))

    if is_log:
        D = np.log10(D)
    return D[0] if one_d else D


forward_anemone.last_calibration = {"mode": None, "k": {}, "residual": {}}


def _forward_varying_height(system, M_c, T_c, tx_height, bin_width, scale,
                            device, progress_callback, showInfo):
    """M_c (nd, K) resistivity, T_c (nd, K-1) per-sounding thicknesses
    (already layer-compressed by the caller)."""
    anemone, torch = _require_anemone()
    from integrate.integrate import _report_progress

    tx_height = np.asarray(tx_height, dtype=float).ravel()
    nd = M_c.shape[0]
    if bin_width and bin_width > 0:
        bin_id = np.round(tx_height / bin_width).astype(np.int64)
        z_of = lambda b: float(b) * bin_width
    else:
        uniq = {v: i for i, v in enumerate(np.unique(tx_height))}
        bin_id = np.array([uniq[v] for v in tx_height], dtype=np.int64)
        z_of = lambda b: float(np.unique(tx_height)[b])

    dev = torch.device(device)
    M_t = torch.as_tensor(M_c, dtype=torch.float64).to(dev)
    T_t = torch.as_tensor(T_c, dtype=torch.float64).to(dev)

    n_used = None
    D = None
    done = 0
    for b in np.unique(bin_id):
        rows = np.where(bin_id == b)[0]
        fwr, slices = _build_forward(system, z_of(b), device)
        with torch.no_grad():
            raw = fwr(M_t[rows].T, T_t[rows].T).detach().cpu().numpy()
        raw = np.atleast_2d(raw)  # anemone squeezes model axis for n_models==1
        cols = [(sc * raw[:, s]) for s, sc in zip(slices, scale)]
        block = np.concatenate(cols, axis=1)
        if D is None:
            n_used = block.shape[1]
            D = np.full((nd, n_used), np.nan, dtype=float)
        D[rows] = block
        done += rows.size
        if progress_callback is not None:
            _report_progress(progress_callback, done, nd, "computing",
                             "Forward modeling (%d/%d soundings)" % (done, nd))
    return D


def prior_data_anemone(f_prior_h5, file_gex=None, N=0, doMakePriorCopy=True,
                       im=1, id=1, im_height=0, is_log=False,
                       altitude_bin_width=1.0, device="cpu",
                       calibration="auto", calibration_reference=None,
                       calibration_factor=None, calibration_tol=0.05,
                       doCompress=True, force_replace=False, f_prior_data_h5="",
                       **kwargs):
    """Generate prior data ``/D{id}`` for the anemone TDEM forward backend.

    Mirrors :func:`integrate.prior_data_gaaem` but loads ``M{im}`` **as
    resistivity** (no ``1/``) and forwards it through :func:`forward_anemone`.
    Returns the path to the prior-data h5 (always the return value).
    """
    import multiprocessing
    import time
    import h5py
    import integrate as ig
    from integrate.integrate import _report_progress

    if multiprocessing.current_process().name != "MainProcess":
        return None

    anemone, _torch = _require_anemone()
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
                    if file_gex else "ANEMONE")
            stem = os.path.splitext(f_prior_h5)[0]
            f_prior_data_h5 = ("%s_%s_N%d_anemone.h5" % (stem, base, N)
                               if N < N_in else
                               "%s_%s_anemone.h5" % (stem, base))
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5, N, showInfo=showInfo)
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
    D = forward_anemone(M=M, thickness=thickness, file_gex=file_gex,
                        tx_height=tx_height,
                        altitude_bin_width=altitude_bin_width, is_log=is_log,
                        device=device, calibration=calibration,
                        calibration_reference=calibration_reference,
                        calibration_factor=calibration_factor,
                        calibration_tol=calibration_tol, doCompress=doCompress,
                        progress_callback=progress_callback, showInfo=showInfo)
    if showInfo > -1:
        dt = time.time() - t1
        print("prior_data_anemone: %.1fs / %d soundings (%.1f ms/sounding)"
              % (dt, M.shape[0], 1000 * dt / max(M.shape[0], 1)))

    _report_progress(progress_callback, N, N, "saving",
                     "Saving forward data to %s" % f_prior_data_h5)

    cal = getattr(forward_anemone, "last_calibration", {"mode": None, "k": {},
                                                        "residual": {}})
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
        a["method"] = "anemone"
        a["type"] = "TDEM"
        a["im"] = im
        a["id"] = id
        a["is_log"] = bool(is_log)
        a["altitude_bin_width"] = float(altitude_bin_width)
        a["device"] = str(device)
        a["calibration"] = str(cal.get("mode") or "gex")
        a["anemone_version"] = str(getattr(anemone, "__version__", "unknown"))
        for name, kval in (cal.get("k") or {}).items():
            a["calibration_factor_%s" % name] = float(kval)
        for name, rval in (cal.get("residual") or {}).items():
            a["calibration_residual_%s" % name] = float(rval)

    ig.integrate_update_prior_attributes(f_prior_data_h5)
    _report_progress(progress_callback, N, N, "completed",
                     "Forward data saved to %s" % f_prior_data_h5)
    return f_prior_data_h5
