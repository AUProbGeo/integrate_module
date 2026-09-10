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
    shared_rcv = m0["receiver"]
    shared_filt = m0["filterfunc"]
    shared_filt_sig = _filter_sig(shared_filt)

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
        fwr = part if fwr is None else (fwr + part)

    if n_moments == 2:
        slices = [fwr.t1slc, fwr.t2slc]
    else:
        slices = [slice(None)]

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
        tx_z = abs(float(np.atleast_1d(G[txkey])[2]))
    else:
        tx_z = 0.0

    rxkey = "RxCoilPosition1" if "RxCoilPosition1" in G else "RxCoilPosition"
    rx_xyz = np.abs(np.atleast_1d(np.asarray(G[rxkey], dtype=float)))

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
        receiver = Receiver(x=float(rx_xyz[0]), y=float(rx_xyz[1]),
                            z=float(rx_xyz[2]))

        wf_key = ("WaveformLM" if name == "LM" else "WaveformHM"
                  if name == "HM" else "WaveformLM")
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

        turns_key = "NumberOfTurnsLM" if name == "LM" else "NumberOfTurnsHM"
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
        "moments": moments,
        "gex_signature": _gex_signature(g),
        "gex": g,
    }


def _used_gate_times(system):
    times = [m["gate_times"] for m in system["moments"]]
    names = [m["name"] or f"CH{i + 1}" for i, m in enumerate(system["moments"])]
    return times, names


def _moment_scale(system, calibration_factor):
    """Deterministic per-moment factor  k * I_approx * n_turns  (k defaults 1)."""
    cf = calibration_factor or {}
    out = []
    for i, m in enumerate(system["moments"]):
        name = m["name"] or f"CH{i + 1}"
        k = float(cf.get(name, cf.get(i, 1.0)))
        out.append(k * m["tx_current"] * m["n_turns"])
    return out  # sign handled in the evaluator


def forward_anemone(M=np.array(()), thickness=np.array(()), file_gex=None,
                    GEX=None, tx_height=np.array(()), altitude_bin_width=1.0,
                    is_log=False, device="cpu", calibration="auto",
                    calibration_reference=None, calibration_factor=None,
                    calibration_tol=0.05, showtime=False, showInfo=0,
                    progress_callback=None, **kwargs):
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
    varying = tx_height.size > 1 and not np.allclose(tx_height, tx_height[0])

    # k_moment: from calibration_factor now; Task 5 fills the fitted path.
    if calibration in ("auto", "gex") and calibration_factor:
        k_by_moment = calibration_factor
    elif calibration == "gex":
        raise ValueError("calibration='gex' needs calibration_factor per moment")
    else:
        k_by_moment = calibration_factor  # may be None -> k=1 (Task 5 overrides)

    scale = _moment_scale(system, k_by_moment)
    thk_t = torch.as_tensor(thickness, dtype=torch.float64)

    t0 = time.time()
    if varying:
        D = _forward_varying_height(system, M, thk_t, tx_height,
                                    altitude_bin_width, scale, device,
                                    progress_callback, showInfo)  # Task 6
    else:
        tx_z = None
        if tx_height.size >= 1:
            tx_z = float(tx_height[0])
        elif not system["has_tx_coil_position"]:
            tx_z = 40.0
        fwr, slices = _build_forward(system, tx_z, device)
        M_t = torch.as_tensor(M, dtype=torch.float64).to(torch.device(device))
        with torch.no_grad():
            raw = fwr(M_t.T, thk_t.to(torch.device(device)))
        raw = np.asarray(raw.detach().cpu().numpy(), dtype=float)
        raw = np.atleast_2d(raw)
        cols = []
        for s, sc in zip(slices, scale):
            cols.append(-sc * raw[:, s])
        D = np.concatenate(cols, axis=1)

    if showtime:
        print("forward_anemone: %.1f ms/model (%d models)"
              % (1000 * (time.time() - t0) / nd, nd))

    if is_log:
        D = np.log10(D)
    return D[0] if one_d else D


def _forward_varying_height(system, M, thk_t, tx_height, bin_width, scale,
                            device, progress_callback, showInfo):
    raise NotImplementedError  # Task 6


def prior_data_anemone(f_prior_h5, file_gex=None, N=0, doMakePriorCopy=True,
                       im=1, id=1, im_height=0, is_log=False,
                       altitude_bin_width=1.0, device="cpu",
                       calibration="auto", calibration_reference=None,
                       calibration_factor=None, calibration_tol=0.05,
                       force_replace=False, f_prior_data_h5="", **kwargs):
    raise NotImplementedError  # Task 7
