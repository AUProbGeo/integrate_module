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


def forward_anemone(M=np.array(()), thickness=np.array(()), file_gex=None,
                    GEX=None, tx_height=np.array(()), altitude_bin_width=1.0,
                    is_log=False, device="cpu", calibration="auto",
                    calibration_reference=None, calibration_factor=None,
                    calibration_tol=0.05, showtime=False, showInfo=0,
                    progress_callback=None, **kwargs):
    raise NotImplementedError  # Tasks 3-6


def prior_data_anemone(f_prior_h5, file_gex=None, N=0, doMakePriorCopy=True,
                       im=1, id=1, im_height=0, is_log=False,
                       altitude_bin_width=1.0, device="cpu",
                       calibration="auto", calibration_reference=None,
                       calibration_factor=None, calibration_tol=0.05,
                       force_replace=False, f_prior_data_h5="", **kwargs):
    raise NotImplementedError  # Task 7
