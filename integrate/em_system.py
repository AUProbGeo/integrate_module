"""
Backend-neutral description of a TDEM system read from an Aarhus ``.gex`` file.

``gex_to_em_system`` turns a GEX (path or the dict returned by
``read_gex_workbench``/``read_gex``) into a plain numpy dict that a forward
modeller can consume: transmitter loop polygon, receiver offset, per-moment
waveform table, low-pass filters and the *used* gate windows.  It follows the
conventions of the GA-AEM/STM path (``write_stm_files`` / ``_stm_lowpass_lists``
in ``integrate_io``) so that any backend built on it is comparable with
``forward_gaaem``:

* gates ``[RemoveInitialGates:NoGates]`` per channel, times shifted by
  ``GateTimeShift + MeaTimeDelay``;
* filters = the channel's ``TiBLowPassFilter`` **plus every**
  ``General.RxCoilLPFilter*`` entry, orders rounded;
* ``rx_dz = Rx_z - Tx_z`` in the GEX frame, where **z is positive down**: a
  SkyTEM ``RxCoilPosition1 z = -2`` means the receiver sits 2 m *above* the
  frame, so a backend must place it at ``tx_height - rx_dz``.  (Validated
  against AarhusInv on SkyTEM: 1.8 %/0.8 % LM/HM with the receiver above vs
  8 %/5 % below.  Note ``forward_gaaem`` passes this value straight to GA-AEM,
  whose ``txrx_dz`` is positive *up*, i.e. it currently puts the SkyTEM
  receiver below the frame; anemone does the same.)

The GA-AEM low-pass transfer function is ``(1/(1 + i f/fc))**order`` (cascaded
first-order poles); ``lowpass_response`` reproduces it.

Used by :mod:`integrate.simpeg_forward`.  :mod:`integrate.anemone_forward`
keeps its own (validated, torch-based) parser and only shares the small
geometry helpers defined here.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["gex_to_em_system", "lowpass_response", "chain_response", "compress_model"]


# --------------------------------------------------------------------------- #
# small helpers (also imported by anemone_forward)
# --------------------------------------------------------------------------- #
def _butter_rows(arr):
    """Normalise an RxCoilLPFilter value to a list of (order, fcut) rows."""
    a = np.atleast_2d(np.asarray(arr, dtype=float))
    if a.shape == (1, 2):
        return [(a[0, 0], a[0, 1])]
    return [(row[0], row[1]) for row in a]


def _polygon_area(x, y):
    """Unsigned shoelace area of the polygon (x, y) (closed or open)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _signed_area(xy):
    x, y = np.asarray(xy, dtype=float).T
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient_loop(xy, clockwise=True):
    """Return the open polygon ``xy`` (n, 2) with the requested winding.

    Shoelace sign convention: counter-clockwise => positive signed area.
    """
    xy = np.asarray(xy, dtype=float)
    ccw = _signed_area(xy) > 0
    if ccw == clockwise:
        xy = xy[::-1].copy()
    return xy


def lowpass_response(f, fcut, order):
    """GA-AEM low-pass transfer function ``(1/(1 + i f/fc))**order``.

    ``f`` in Hz (array-like), ``fcut`` in Hz.  For ``order == 1`` this equals a
    first-order Butterworth; for higher orders GA-AEM cascades first-order
    poles (anemone uses Butterworth polynomials there, so the two differ).
    """
    f = np.asarray(f, dtype=float)
    return (1.0 / (1.0 + 1j * f / float(fcut))) ** float(order)


def chain_response(f, filters):
    """Product of ``lowpass_response`` over ``filters = [(order, fcut), ...]``."""
    f = np.asarray(f, dtype=float)
    out = np.ones(f.shape, dtype=complex)
    for order, fcut in filters:
        out = out * lowpass_response(f, fcut, order)
    return out


def compress_model(rho, thickness):
    """Merge adjacent layers with identical resistivity (single model).

    Same rule as ``forward_gaaem``'s ``doCompress`` / anemone's
    ``_compress_batch``: a run of equal adjacent values becomes one layer whose
    thickness is the sum.  The last (half-space) layer has no thickness.

    Returns ``(rho_c (k,), thk_c (k-1,))``.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    thickness = np.asarray(thickness, dtype=float).ravel()
    if rho.size != thickness.size + 1:
        raise ValueError("compress_model: len(rho) must equal len(thickness) + 1")
    cz = np.concatenate([[0.0], np.cumsum(thickness)])
    edges = np.concatenate([[0], np.where(np.diff(rho) != 0)[0] + 1])
    return rho[edges], np.diff(cz[edges])


# --------------------------------------------------------------------------- #
# GEX -> system dict
# --------------------------------------------------------------------------- #
def _load_gex_dict(gex):
    if isinstance(gex, dict):
        return gex
    import integrate as ig
    try:
        return ig.read_gex_workbench(gex)
    except (ValueError, KeyError):
        return ig.read_gex(gex)


def _first(v, default=None):
    if v is None:
        return default
    return float(np.atleast_1d(np.asarray(v, dtype=float))[0])


def _waveform(G, moment):
    """(n, 2) waveform table for ``moment`` in {'LM', 'HM'}."""
    key = f"Waveform{moment}"
    wf = G.get(key)
    if wf is None:
        keys = sorted((k for k in G if k.startswith(f"{key}Point")),
                      key=lambda k: int(k.replace(f"{key}Point", "") or "0"))
        if not keys:
            raise KeyError(f"GEX has no {key} / {key}Point## entries")
        wf = np.array([np.atleast_1d(G[k])[:2] for k in keys], dtype=float)
    wf = np.asarray(wf, dtype=float)
    if wf.ndim != 2 or wf.shape[1] < 2:
        raise ValueError(f"{key}: expected an (n, 2) table, got shape {wf.shape}")
    return wf[:, :2]


def _loop_xy(G):
    pts = G.get("TxLoopPoint")
    if pts is None:
        keys = sorted((k for k in G if k.startswith("TxLoopPoint")),
                      key=lambda k: int(k.replace("TxLoopPoint", "") or "0"))
        if not keys:
            raise KeyError("GEX has no TxLoopPoint / TxLoopPoint## entries")
        pts = np.array([np.atleast_1d(G[k])[:2] for k in keys], dtype=float)
    pts = np.asarray(pts, dtype=float)[:, :2]
    if pts.shape[0] > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def gex_to_em_system(gex, showInfo=0):
    """
    Parse a GEX into a backend-neutral system description.

    Parameters
    ----------
    gex : str or dict
        Path to a ``.gex`` file, or the dict from ``read_gex_workbench``.
    showInfo : int
        Verbosity.

    Returns
    -------
    dict with keys
        ``n_moments``, ``has_tx_coil_position``, ``tx_z`` (m above ground of
        the Tx coil for ground systems, else 0), ``rx_xy`` (x, y offset from
        the Tx frame), ``rx_dz`` (``Rx_z - Tx_z`` in the GEX z-down frame;
        receiver height = ``tx_height - rx_dz``), ``loop_xy`` ((n+1, 2) closed, clockwise),
        ``tx_area_poly``, ``tx_area_gex``, ``moments`` (list of dicts:
        ``name, waveform (n,2), filters [(order, fcut)...], gate_centre,
        gate_open, gate_close, n_turns, tx_current, rep_freq``) and
        ``gex_signature``.

    Raises
    ------
    ValueError
        If any used gate opens at ``t <= 0`` (SimPEG would silently zero it).
    NotImplementedError
        For a non-Z receiver polarisation.
    """
    gd = _load_gex_dict(gex)
    G = gd["General"]

    has_txpos = any(k.startswith("TxCoilPosition") for k in G)
    if has_txpos:
        txkey = "TxCoilPosition1" if "TxCoilPosition1" in G else "TxCoilPosition"
        tx_pos = np.atleast_1d(np.asarray(G[txkey], dtype=float))
        tx_pos_z = float(tx_pos[2])
        tx_z = abs(tx_pos_z)
    else:
        tx_pos_z = 0.0
        tx_z = 0.0
    rxkey = "RxCoilPosition1" if "RxCoilPosition1" in G else "RxCoilPosition"
    rx = np.atleast_1d(np.asarray(G[rxkey], dtype=float))
    rx_xy = (float(rx[0]), float(rx[1]))
    if has_txpos:
        rx_xy = (float(rx[0] - tx_pos[0]), float(rx[1] - tx_pos[1]))
    rx_dz = float(rx[2]) - tx_pos_z  # GEX z is positive down (SkyTEM: -2 => 2 m above the frame)

    loop = _orient_loop(_loop_xy(G), clockwise=True)
    tx_area_poly = _polygon_area(loop[:, 0], loop[:, 1])
    tx_area_gex = float(G.get("TxLoopArea", tx_area_poly))
    if abs(tx_area_poly - tx_area_gex) / max(tx_area_gex, 1e-9) > 0.05:
        print(f"gex_to_em_system: WARNING polygon area {tx_area_poly:.3g} m^2 differs "
              f"from TxLoopArea {tx_area_gex:.3g} m^2; using the polygon.")
    loop_closed = np.vstack([loop, loop[:1]])

    rx_rows = []
    for key in sorted(k for k in G if k.startswith("RxCoilLPFilter")):
        rx_rows.extend(_butter_rows(G[key]))

    gates = np.asarray(G["GateArray"], dtype=float)
    if gates.ndim != 2 or gates.shape[1] < 3:
        raise ValueError("GateArray must be (n, 3) = (centre, open, close)")

    n_moments = sum(1 for k in gd if k.startswith("Channel"))
    moments = []
    for ch in range(1, n_moments + 1):
        chan = gd.get(f"Channel{ch}", {})
        pol = str(chan.get("ReceiverPolarizationXYZ", "Z")).strip().upper()
        if pol and pol != "Z":
            raise NotImplementedError(
                f"Channel{ch}: receiver polarisation {pol!r} not supported (Z only)")
        mom = "LM" if ch == 1 else "HM"
        name = str(chan.get("TransmitterMoment", mom)).strip() or mom

        i0 = int(_first(chan.get("RemoveInitialGates"), 0))
        i1 = int(_first(chan.get("NoGates"), gates.shape[0]))
        shift = _first(chan.get("GateTimeShift"), 0.0) + _first(chan.get("MeaTimeDelay"), 0.0)
        win = gates[i0:i1, :3] + shift
        if win.shape[0] == 0:
            raise ValueError(f"Channel{ch}: no gates left after RemoveInitialGates/NoGates")
        if np.any(win[:, 1] <= 0):
            raise ValueError(
                f"Channel{ch}: {int(np.sum(win[:, 1] <= 0))} used gate(s) open at t <= 0 "
                "after GateTimeShift; increase RemoveInitialGates")

        filters = []
        tib = chan.get("TiBLowPassFilter")
        if tib is not None:
            tib = np.atleast_1d(np.asarray(tib, dtype=float))
            filters.append((int(round(tib[0])), float(tib[1])))
        filters.extend((int(round(o)), float(fc)) for o, fc in rx_rows)

        moments.append({
            "name": name,
            "waveform": _waveform(G, mom),
            "filters": filters,
            "gate_centre": win[:, 0].copy(),
            "gate_open": win[:, 1].copy(),
            "gate_close": win[:, 2].copy(),
            "n_turns": _first(G.get(f"NumberOfTurns{mom}"), 1.0),
            "tx_current": _first(chan.get("TxApproximateCurrent"), 1.0),
            "rep_freq": _first(chan.get("RepFreq"), 0.0),
        })

    sig_src = repr({k: (np.asarray(v).tolist() if hasattr(v, "shape") else v)
                    for k, v in sorted(G.items())}) + repr(
        [{k: (np.asarray(v).tolist() if hasattr(v, "shape") else v)
          for k, v in sorted(gd.get(f"Channel{ch}", {}).items())}
         for ch in range(1, n_moments + 1)])
    system = {
        "n_moments": n_moments,
        "has_tx_coil_position": bool(has_txpos),
        "tx_z": float(tx_z),
        "rx_xy": rx_xy,
        "rx_dz": float(rx_dz),
        "loop_xy": loop_closed,
        "tx_area_poly": float(tx_area_poly),
        "tx_area_gex": float(tx_area_gex),
        "moments": moments,
        "gex_signature": hashlib.md5(sig_src.encode()).hexdigest(),
    }
    if showInfo:
        print(f"gex_to_em_system: {n_moments} moment(s), loop area {tx_area_poly:.3g} m^2, "
              f"tx_z={tx_z:.2f} m, rx offset {rx_xy} dz={rx_dz:+.2f} m, gates "
              + " + ".join(str(m['gate_centre'].size) for m in moments))
    return system


def used_gate_count(system):
    """Total number of data columns (sum of used gates over moments)."""
    return int(sum(m["gate_centre"].size for m in system["moments"]))
