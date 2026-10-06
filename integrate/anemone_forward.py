"""anemone (PyTorch) 1D TDEM forward modelling as an alternative to GA-AEM.

Public API mirrors ``forward_gaaem`` / ``prior_data_gaaem`` in
``integrate.integrate``.  ``anemone`` and ``torch`` are imported lazily so
importing ``integrate`` never requires them.

See docs/superpowers/specs/2026-09-10-anemone-forward-backend-design.md
"""
from __future__ import annotations

import os

import numpy as np

from integrate.em_system import (_butter_rows, _polygon_area,  # shared helpers
                                 _rx_coil_filter_by_coil, rx_coil_lowpass)

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
    """Return a libaarhusxyz GEX object, parsing a path or wrapping a dict.

    libaarhusxyz.GEX prints one "header [...] parsed" line per section; that
    chatter is swallowed here since it cannot be turned off in the library.
    """
    if isinstance(gex, str):
        try:
            import contextlib, io
            from libaarhusxyz import GEX
            with contextlib.redirect_stdout(io.StringIO()):
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
    """Return signature (fcut, order, damping) tuples for a FilterChain (None -> [])."""
    if fc is None:
        return []
    return [(round(float(f.fcut), 3), int(f.order),
             round(float(getattr(f, "damping", 0.0)), 6)) for f in fc]


class _DampedLowPass:
    """Second-order low-pass ``1/(1 + 2 zeta s + s^2)``, ``s = i f/fcut``.

    Same interface as ``anemone.system.ButterworthFilter`` (``__call__(f)`` in
    Hz, ``omega_filter(omega)``, ``fcut``, ``order``), but with a free damping
    ``zeta``; anemone's order-2 Butterworth is the special case zeta = 1/sqrt(2).
    Used for the receiver-coil response with ``rx_coil_filter='damped2'``.
    """

    def __init__(self, fcut, damping):
        self.fcut = float(fcut)
        self.damping = float(damping)
        self.order = 2

    def __call__(self, f):
        s = 1j * f / self.fcut
        return 1.0 / (1.0 + 2.0 * self.damping * s + s * s)

    def omega_filter(self, omega):
        return self(omega / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# System response (.sr2) support
# ---------------------------------------------------------------------------

def read_sr2(file_sr2):
    """Read a Workbench/AarhusInv ``.sr2`` system-response file.

    Format: ``//`` comment lines, then one or more blocks of a header line
    ``channel# #repetitions #npts t_begin(yyyy mm dd hh nn ss zzz) t_end(...)``
    followed by ``npts`` rows ``time [s]  dSR/dt``.  The samples are the
    second time-derivative of the (unit-current) transmitter waveform with the
    system's own filtering folded in, i.e. they take the place of anemone's
    ``Waveform.d2wdt2`` (after multiplying by the sample spacing).

    Returns
    -------
    dict
        ``{channel: {"time", "dsrdt", "n_rep", "header"}}`` with numpy arrays.
    """
    with open(file_sr2, "r") as f:
        lines = [ln.split("//")[0].strip() for ln in f]
    lines = [ln for ln in lines if ln]
    out = {}
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) < 3:
            raise ValueError(f"read_sr2: bad block header in {file_sr2}: "
                             f"'{lines[i]}'")
        ch, n_rep, npts = int(head[0]), int(head[1]), int(head[2])
        rows = np.array([lines[j].split()[:2]
                         for j in range(i + 1, i + 1 + npts)], dtype=float)
        if rows.shape[0] != npts:
            raise ValueError(f"read_sr2: channel {ch} expects {npts} rows, "
                             f"found {rows.shape[0]} in {file_sr2}")
        out[ch] = {"time": rows[:, 0], "dsrdt": rows[:, 1], "n_rep": n_rep,
                   "header": lines[i]}
        i += 1 + npts
    if not out:
        raise ValueError(f"read_sr2: no data blocks in {file_sr2}")
    return out


def _make_waveform(time, weights, amplitude, torch):
    """anemone ``Waveform``-compatible object with explicit convolution weights.

    anemone only uses ``.time``, ``.amplitude`` and ``.d2wdt2`` (the impulse
    weights convolved with the B step response), so the weights are set
    directly instead of being derived from a piecewise-linear current.
    """
    from anemone.system import Waveform
    wf = Waveform.__new__(Waveform)
    wf.time = torch.as_tensor(np.asarray(time, dtype=float), dtype=torch.float64)
    wf.amplitude = torch.as_tensor(np.asarray(amplitude, dtype=float),
                                   dtype=torch.float64)
    wf.d2wdt2 = torch.as_tensor(np.asarray(weights, dtype=float),
                                dtype=torch.float64)
    return wf


def _sr2_to_waveform(t, dsrdt, torch):
    """Convert sampled dSR/dt to an anemone waveform with trapezoid weights.

    ``anemone.system.SystemResponse`` uses backward-difference weights
    ``dSR/dt[j] * (t[j]-t[j-1])``, which shifts each sample half a step late;
    trapezoid weights ``dSR/dt[j] * (t[j+1]-t[j-1])/2`` are centred.
    """
    t = np.asarray(t, dtype=float)
    dsrdt = np.asarray(dsrdt, dtype=float)
    dt = np.diff(t)
    w = np.zeros_like(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    weights = dsrdt * w
    # Recovered unit-normalised current (diagnostics only).
    didt = np.cumsum(weights)
    current = np.concatenate([[0.0], np.cumsum(0.5 * (didt[1:] + didt[:-1]) * dt)])
    return _make_waveform(t, weights, current, torch)


def _waveform_arrays(wf):
    return (wf.time.detach().cpu().numpy().astype(float),
            wf.d2wdt2.detach().cpu().numpy().astype(float),
            wf.amplitude.detach().cpu().numpy().astype(float))


def _add_previous_pulses(wf, rep_freq, sign_pattern, n, torch):
    """Prepend ``n`` earlier pulses (period ``1/(2 rep_freq)``) to ``wf``.

    Pulse ``k`` back in time is shifted by ``-k/(2 rep_freq)`` and carries the
    relative polarity from ``sign_pattern`` (``[1, -1]`` -> alternating), as for
    a bipolar SkyTEM/tTEM transmitter.  The returned waveform's convolution
    weights are the superposition of all pulses.
    """
    if not n or n <= 0:
        return wf
    if not rep_freq or rep_freq <= 0:
        raise ValueError("n_previous_pulses needs a positive RepFreq in the GEX")
    pat = np.atleast_1d(np.asarray(sign_pattern if sign_pattern is not None
                                   else [1.0, -1.0], dtype=float))
    t, w, a = _waveform_arrays(wf)
    t_half = 0.5 / float(rep_freq)
    ts, ws = [t], [w]
    for k in range(1, int(n) + 1):
        sign = pat[(-k) % len(pat)] * pat[0]
        ts.insert(0, t - k * t_half)
        ws.insert(0, sign * w)
    t_all = np.concatenate(ts)
    w_all = np.concatenate(ws)
    order = np.argsort(t_all, kind="stable")
    # amplitude is diagnostic only; keep the current pulse's, zero elsewhere
    a_all = np.concatenate([np.zeros(t.size * int(n)), a])[order]
    return _make_waveform(t_all[order], w_all[order], a_all, torch)


def _resolve_sr2(gex, g, file_sr2):
    """Return the SR2 path to use (or None).

    Explicit ``file_sr2`` wins.  Otherwise, if any channel of the GEX sets
    ``SystemResponseConvolution=1`` and the GEX was given as a path, look for
    ``<gex stem>.sr2`` next to it; a missing file is an error since the GEX
    waveform is then usually a placeholder.
    """
    if file_sr2:
        if not os.path.isfile(file_sr2):
            raise FileNotFoundError(f"file_sr2={file_sr2} does not exist")
        return file_sr2
    wants_sr = any(
        float(np.atleast_1d(v.get("SystemResponseConvolution", 0))[0]) == 1
        for k, v in g.gex_dict.items()
        if k.startswith("Channel") and isinstance(v, dict))
    if not wants_sr:
        return None
    if isinstance(gex, str):
        cand = os.path.splitext(gex)[0] + ".sr2"
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "GEX sets SystemResponseConvolution=1 but no system-response file was "
        "found; pass file_sr2=... (looked for '<gex stem>.sr2' next to the GEX)")


def _file_md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# Floor for the transform time grid (s); see _build_forward.
_TMIN_FLOOR = 1e-8


def _grid_tmin(times, wf):
    """Smallest positive gate-minus-waveform delay, floored at _TMIN_FLOOR."""
    d = (np.asarray(times, dtype=float)[:, None]
         - wf.time.detach().cpu().numpy()[None, :])
    w = wf.d2wdt2.detach().cpu().numpy()
    d = d[(d > 0) & (w[None, :] != 0)]
    if d.size == 0:
        return None
    return max(float(d.min()), _TMIN_FLOOR)


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
    # The Rx coil position in the GEX is an OFFSET from the Tx frame in the
    # GEX frame, where z is positive DOWN (SkyTEM RxCoilPosition z=-2: Rx 2 m
    # above the frame).  anemone's z is height above ground (positive up), so
    # receiver_z = loop_z - dz.  anemone uses sfield_dz = |srcz| + |rz| and
    # pfield_dz = |srcz - rz|, so both must be positive heights above ground.
    # Validated against AarhusInv on SkyTEM (see ISSUE_rx_z_sign.md).
    loop_z = float(shared_src.z[0])
    dx, dy, _dz = system.get("rx_offset", (0.0, 0.0, 0.0))
    receiver_z = loop_z - float(system.get("rx_offset_z_rel_tx", 0.0))
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
        # 'boxcar': anemone averages dB/dt over [gate open, gate close] when
        # given the open times as `times` and the close times as `t_end`.
        if system.get("gate_integration") == "boxcar":
            t_first = m["gate_open"]
            t_end = torch.as_tensor(m["gate_close"], dtype=torch.float64).to(dev)
        else:
            t_first = m["gate_times"]
            t_end = None
        times = torch.as_tensor(t_first, dtype=torch.float64).to(dev)
        # anemone puts the floor of its transform time grid at min(gate)/2 and
        # only lowers it when the whole waveform precedes the first gate.  The
        # convolution sum_j w_j Bstep(t - t_j) silently drops any term whose
        # delay t - t_j falls below that floor (waveform/SR samples just before
        # an early gate, e.g. a turn-off tail past t=0).  Lower the floor to
        # cover the smallest delay before building the transform.
        dmin = _grid_tmin(t_first, wf)
        part = Forward(src, rcv, times, wf, filt, t_end=t_end,
                       tolerance=system.get("tolerance", 1e-6),
                       do_setup=False)
        if dmin is not None and dmin / 2. < float(part.tmin):
            part.tmin = torch.as_tensor(dmin / 2., dtype=torch.float64).to(dev)
        part.setup()
        part.device = dev
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


def gex_to_anemone_system(gex, showInfo=0, file_sr2=None, sr_filters=False,
                          n_previous_pulses=0, rx_coil_filter="two_pole",
                          gate_integration="centre", tolerance=1e-6):
    """Build the anemone system description from a GEX (+ optional SR2).

    rx_coil_filter : {'two_pole', 'damped2', 'cascade'}
        How the GEX ``RxCoilLPFilter`` entries are modelled.  All readings but
        ``'cascade'`` use only the entry of the channel's ``RxCoilNumber``,
        ``zeta fcut``, as the second-order coil response
        ``1/(1 + 2 zeta s + s^2)`` (see ``ISSUE_rx_coil_filter.md``):

        * ``'two_pole'`` (default): two first-order poles at ``fcut/zeta``
          (:func:`integrate.em_system.rx_coil_lowpass`), the same filters
          GA-AEM and SimPEG use;
        * ``'damped2'``: the exact second-order response;
        * ``'cascade'``: the earlier reading, every entry as a Butterworth
          filter of order ``round(first value)``; kept for comparisons.
    gate_integration : {'centre', 'boxcar'}
        ``'centre'`` (default) evaluates dB/dt at the gate centre time;
        ``'boxcar'`` averages it over the gate (open to close), using anemone's
        built-in gating.
    tolerance : float or None
        anemone's kernel pruning (default ``1e-6``).  At setup, for each gate
        the (frequency, wavenumber) terms are ranked by their contribution for
        a 100 ohm-m half-space, and the smallest terms adding up to at most
        ``tolerance`` of the gate's total absolute contribution are dropped.
        Fewer terms = faster evaluation (tTEM GEX, GPU: ``1e-5`` ~1.4x,
        ``1e-4`` ~1.7x faster than ``1e-6``).  On rough random 8-layer models,
        the share of gates within 1 % of the unpruned result is 99.4 % for
        ``1e-6``, 96 % for ``1e-5``, 86 % for ``1e-4`` and 76 % for ``1e-3``;
        the large errors are on gates far below the sounding's peak.
        ``None`` keeps all terms (~70x more; needs a small ``batch_size`` on
        GPU).  See ``FORWARD_MODELS.md``.
    file_sr2 : str, optional
        Workbench ``.sr2`` system-response file.  If omitted and the GEX sets
        ``SystemResponseConvolution=1``, ``<gex stem>.sr2`` next to the GEX is
        used (error if missing).  Channels with an SR block use it instead of
        the GEX waveform.
    sr_filters : bool
        Also apply the GEX Butterworth low-pass filters to SR channels.  Off by
        default: the measured system response already contains the system's
        filtering.
    n_previous_pulses : int
        Number of earlier bipolar pulses (``RepFreq``/``SignPattern`` from the
        GEX channel) superposed on the waveform (default 0 = single pulse).
    """
    anemone, torch = _require_anemone()
    from anemone.system import (Loop, Receiver, Waveform, FilterChain,
                                ButterworthFilter)

    if rx_coil_filter not in ("two_pole", "damped2", "cascade"):
        raise ValueError("rx_coil_filter must be 'two_pole', 'damped2' or 'cascade', "
                         "not %r" % (rx_coil_filter,))
    if tolerance is not None and not 0 < float(tolerance) < 1:
        raise ValueError("tolerance must be None or between 0 and 1, not %r" % (tolerance,))
    if gate_integration not in ("centre", "boxcar"):
        raise ValueError("gate_integration must be 'centre' or 'boxcar', not %r"
                         % (gate_integration,))

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
    # an absolute height.  Kept in the GEX frame (z positive down);
    # rx_offset_z_rel_tx > 0 means the receiver is BELOW the transmitter.
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
    rx_by_coil = _rx_coil_filter_by_coil(G)

    sr2_path = _resolve_sr2(gex, g, file_sr2)
    sr = read_sr2(sr2_path) if sr2_path else {}
    if sr2_path and showInfo > 0:
        print(f"gex_to_anemone_system: system response from {sr2_path} "
              f"(channels {sorted(sr)})")

    moments = []
    for ch in range(1, n_moments + 1):
        chan = g.gex_dict.get(f"Channel{ch}", {})
        name = str(chan.get("TransmitterMoment",
                            "LM" if ch == 1 else "HM")).strip() or None

        zs = np.full_like(xs, tx_z)
        source = Loop(torch.tensor(xs), torch.tensor(ys), torch.tensor(zs))
        # Placeholder receiver; the real one is rebuilt per tx_z in
        # _build_forward() at loop_z - rx_offset_z_rel_tx.
        receiver = Receiver(x=float(rx_offset[0]), y=float(rx_offset[1]),
                            z=float(tx_z - rx_offset_z_rel_tx))

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
        use_sr = ch in sr and (
            bool(file_sr2) or float(np.atleast_1d(
                chan.get("SystemResponseConvolution", 0))[0]) == 1)
        if use_sr:
            waveform = _sr2_to_waveform(sr[ch]["time"], sr[ch]["dsrdt"], torch)
        else:
            wf = np.asarray(wf, dtype=float)
            waveform = Waveform(torch.tensor(wf[:, 0]), torch.tensor(wf[:, 1]))
        if n_previous_pulses:
            waveform = _add_previous_pulses(
                waveform, float(np.atleast_1d(chan.get("RepFreq", 0))[0]),
                chan.get("SignPattern"), n_previous_pulses, torch)

        filt = []
        tib = chan.get("TiBLowPassFilter")
        if tib is not None:
            tib = np.atleast_1d(np.asarray(tib, dtype=float))
            filt.append(ButterworthFilter(float(tib[1]), int(round(tib[0]))))
        coil = int(np.atleast_1d(chan.get("RxCoilNumber", 1))[0])
        if rx_coil_filter == "two_pole":
            for order, fcut in rx_coil_lowpass(G, coil):
                filt.append(ButterworthFilter(float(fcut), int(order)))
        elif rx_coil_filter == "damped2":
            if coil in rx_by_coil:
                damping, fcut = rx_by_coil[coil]
                filt.append(_DampedLowPass(fcut, damping))
        else:
            for order, fcut in rx_rows:
                filt.append(ButterworthFilter(float(fcut), int(round(order))))
        filterfunc = FilterChain(filt) if filt else None
        if use_sr and not sr_filters:
            filterfunc = None

        i0 = int(g.remove_initial_gates(ch))
        i1 = int(g.no_gates(ch))
        gate_table = np.asarray(g.gate_times(ch), dtype=float)[i0:i1]
        gate_times = gate_table[:, 0]

        turns_key = "NumberOfTurnsLM" if mom == "LM" else "NumberOfTurnsHM"
        moments.append({
            "name": name,
            "source": source,
            "receiver": receiver,
            "waveform": waveform,
            "filterfunc": filterfunc,
            "gate_times": gate_times,
            "gate_open": gate_table[:, 1] if gate_table.shape[1] > 2 else None,
            "gate_close": gate_table[:, 2] if gate_table.shape[1] > 2 else None,
            "n_turns": float(np.atleast_1d(G.get(turns_key, 1))[0]),
            "tx_current": float(np.atleast_1d(
                chan.get("TxApproximateCurrent", 1.0))[0]),
            "tx_area": tx_area_poly,
            "front_gate_delay": float(np.atleast_1d(
                G.get("FrontGateDelay", 0.0))[0]),
            "system_response": bool(use_sr),
        })

    signature = _gex_signature(g)
    if sr2_path or sr_filters or n_previous_pulses:
        signature = "%s|sr=%s|srf=%d|npp=%d" % (
            signature, _file_md5(sr2_path) if sr2_path else "",
            int(bool(sr_filters)), int(n_previous_pulses or 0))
    if rx_coil_filter != "two_pole" or gate_integration != "centre":
        signature = "%s|rxf=%s|gi=%s" % (signature, rx_coil_filter, gate_integration)
    if tolerance != 1e-6:
        signature = "%s|tol=%r" % (signature, tolerance)

    if gate_integration == "boxcar" and any(m["gate_open"] is None for m in moments):
        raise ValueError("gate_integration='boxcar' needs gate open/close times in the GEX")

    return {
        "n_moments": n_moments,
        "gate_integration": gate_integration,
        "tolerance": tolerance,
        "has_tx_coil_position": bool(has_txpos),
        "rx_offset": (float(rx_offset[0]), float(rx_offset[1]),
                      float(rx_offset[2])),
        "rx_offset_z_rel_tx": float(rx_offset_z_rel_tx),
        "moments": moments,
        "gex_signature": signature,
        "gex": g,
        "sr2_path": sr2_path,
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


def _run_batched(fwr, M_t, T_t, batch_size, torch):
    """Run ``fwr`` over the model axis of ``M_t``/``T_t`` in chunks.

    anemone's intermediates scale linearly with the number of models
    (models x layers x frequencies x Hankel-filter length, complex128), so a
    single call with many thousands of soundings can exhaust GPU memory even
    though the input itself is small. Chunking bounds the peak memory at
    roughly ``batch_size`` models regardless of the total.

    M_t: (nd, K) resistivity, T_t: (nd, K-1) thicknesses (both torch tensors
    already on the target device). Returns raw as an (nd, n_gates) ndarray.
    """
    nd = M_t.shape[0]
    if not batch_size or batch_size <= 0 or batch_size >= nd:
        batch_size = nd
    blocks = []
    with torch.no_grad():
        for i0 in range(0, nd, batch_size):
            i1 = min(i0 + batch_size, nd)
            raw = fwr(M_t[i0:i1].T, T_t[i0:i1].T).detach().cpu().numpy()
            blocks.append(np.atleast_2d(raw))  # anemone squeezes for n_models==1
            del raw
    return np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]


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
                    showInfo=0, progress_callback=None, batch_size=1000,
                    file_sr2=None, sr_filters=False, n_previous_pulses=0,
                    rx_coil_filter="two_pole", gate_integration="centre",
                    tolerance=1e-6, **kwargs):
    """Forward TDEM data for one or more resistivity models with anemone.

    batch_size : int, optional
        Number of soundings sent through anemone per call (default 1000).
        Bounds peak (GPU) memory; ``0``/``None`` forwards everything at once.
    file_sr2, sr_filters, n_previous_pulses
        System-response / pulse options, see :func:`gex_to_anemone_system`.
        A GEX with ``SystemResponseConvolution=1`` picks up ``<gex stem>.sr2``
        automatically.
    rx_coil_filter, gate_integration
        Receiver-coil filter model and gate integration, see
        :func:`gex_to_anemone_system`.
    tolerance : float or None
        Kernel pruning tolerance (default ``1e-6``); larger is faster and less
        accurate, see :func:`gex_to_anemone_system`.
    """
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
                                   showInfo=showInfo, file_sr2=file_sr2,
                                   sr_filters=sr_filters,
                                   n_previous_pulses=n_previous_pulses,
                                   rx_coil_filter=rx_coil_filter,
                                   gate_integration=gate_integration,
                                   tolerance=tolerance)

    forward_anemone.last_sr2 = system.get("sr2_path")

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
                                    progress_callback, showInfo,
                                    batch_size)  # Task 6
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
        raw = np.asarray(_run_batched(fwr, M_t, T_t, batch_size, torch),
                         dtype=float)
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
forward_anemone.last_sr2 = None


def _forward_varying_height(system, M_c, T_c, tx_height, bin_width, scale,
                            device, progress_callback, showInfo,
                            batch_size=1000):
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
        raw = _run_batched(fwr, M_t[rows], T_t[rows], batch_size, torch)
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
                       batch_size=1000, randomize=True, file_sr2=None,
                       sr_filters=False, n_previous_pulses=0,
                       rx_coil_filter="two_pole", gate_integration="centre",
                       tolerance=1e-6, **kwargs):
    """Generate prior data ``/D{id}`` for the anemone TDEM forward backend.

    Mirrors :func:`integrate.prior_data_gaaem` but loads ``M{im}`` **as
    resistivity** (no ``1/``) and forwards it through :func:`forward_anemone`.
    ``batch_size`` soundings are forwarded per anemone call (bounds peak GPU
    memory; ``0`` forwards all ``N`` at once). ``randomize`` controls whether a
    copy with ``N < N_in`` draws ``N`` random realizations (default) or the
    first ``N`` sequentially.  ``rx_coil_filter``, ``gate_integration`` and
    ``tolerance`` are passed to :func:`forward_anemone`.
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
    D = forward_anemone(M=M, thickness=thickness, file_gex=file_gex,
                        tx_height=tx_height,
                        altitude_bin_width=altitude_bin_width, is_log=is_log,
                        device=device, calibration=calibration,
                        calibration_reference=calibration_reference,
                        calibration_factor=calibration_factor,
                        calibration_tol=calibration_tol, doCompress=doCompress,
                        batch_size=batch_size,
                        progress_callback=progress_callback, showInfo=showInfo,
                        file_sr2=file_sr2, sr_filters=sr_filters,
                        n_previous_pulses=n_previous_pulses,
                        rx_coil_filter=rx_coil_filter,
                        gate_integration=gate_integration,
                        tolerance=tolerance)
    if showInfo > -1:
        dt = time.time() - t1
        n_sound = M.shape[0]
        dev_label = "cpu" if str(device).split(":")[0] == "cpu" else "gpu"
        print("prior_data_anemone[%s]: Time=%5.1fs/%d soundings. %4.1fms/sounding, %3.1fit/s"
              % (dev_label, dt, n_sound, 1000 * dt / max(n_sound, 1), n_sound / max(dt, 1e-12)))

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
        a["batch_size"] = int(batch_size or 0)
        a["calibration"] = str(cal.get("mode") or "gex")
        a["anemone_version"] = str(getattr(anemone, "__version__", "unknown"))
        sr2_used = getattr(forward_anemone, "last_sr2", None)
        if sr2_used:
            a["sr2"] = str(sr2_used)
            a["sr_filters"] = bool(sr_filters)
        a["n_previous_pulses"] = int(n_previous_pulses or 0)
        a["rx_coil_filter"] = str(rx_coil_filter)
        a["gate_integration"] = str(gate_integration)
        a["tolerance"] = "none" if tolerance is None else float(tolerance)
        for name, kval in (cal.get("k") or {}).items():
            a["calibration_factor_%s" % name] = float(kval)
        for name, rval in (cal.get("residual") or {}).items():
            a["calibration_residual_%s" % name] = float(rval)

    ig.integrate_update_prior_attributes(f_prior_data_h5)
    _report_progress(progress_callback, N, N, "completed",
                     "Forward data saved to %s" % f_prior_data_h5)
    return f_prior_data_h5
