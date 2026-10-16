"""
Combine a Workbench GEX file and a system-response (``.sr2``) file into one GEX.

anemone convolves the B-field step response with the system response directly
(``forward_anemone(file_gex=..., file_sr2=...)``).  GA-AEM and SimPEG only know a
piecewise-linear transmitter current plus simple low-pass filters, so they cannot
read an SR2.  Because the system is linear, convolving the current with the system
response gives the same dB/dt as convolving the response with it: the effective
current ``I(t) = double integral of SR(t)`` (which already contains the
transmitter and receiver filtering) can be written as the GEX waveform points,
with ``SystemResponseConvolution=0`` and the filters switched off (so they are not
applied twice).  The result is a plain GEX that every backend can use.

Limitations: the current is decimated to a piecewise-linear table (error bounded by
``tol``), and the effect of earlier bipolar pulses (anemone ``n_previous_pulses``)
is not represented.
"""

import os
import re

import numpy as np

_FILTER_OFF_FCUT = 1.0e9   # Hz; a low-pass this high is invisible at us gate times


def _read_sr2_blocks(file_sr2):
    """``{channel: {"time", "dsrdt"}}`` from a ``.sr2`` file (no anemone/torch needed)."""
    with open(file_sr2, 'r') as f:
        lines = []
        for ln in f:
            ln = ln.split('//')[0].strip()
            if ln:
                lines.append(ln)
    out = {}
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) < 3:
            raise ValueError("sr2_current: bad block header in %s: '%s'" % (file_sr2, lines[i]))
        ch = int(head[0])
        npts = int(head[2])
        t = np.zeros(npts)
        d = np.zeros(npts)
        for k in range(npts):
            row = lines[i + 1 + k].split()
            t[k] = float(row[0])
            d[k] = float(row[1])
        out[ch] = {'time': t, 'dsrdt': d}
        i += 1 + npts
    if not out:
        raise ValueError("sr2_current: no data blocks in %s" % file_sr2)
    return out


def sr2_effective_current(time, dsrdt, tol=1e-5, tail_tol=1e-4):
    """Piecewise-linear unit current equivalent to a system response.

    Uses the same trapezoid weights as the anemone backend
    (``dSR/dt * (t[j+1]-t[j-1])/2``), integrates twice to the current, removes the
    tiny integration residual after turn-off, and decimates the result with
    Ramer-Douglas-Peucker so that the piecewise-linear current stays within ``tol``
    (absolute, current normalised to ~1) of the sampled one.

    Parameters
    ----------
    time, dsrdt : array_like
        SR2 samples (time [s], second derivative of the unit current).
    tol : float
        Maximum deviation of the decimated current from the sampled current.
    tail_tol : float
        After the last time the current exceeds this value it is forced to zero.

    Returns
    -------
    ndarray (n, 2)
        ``[time, current]`` waveform points, ending with exactly zero current.
    """
    t = np.asarray(time, dtype=float)
    d = np.asarray(dsrdt, dtype=float)
    dt = np.diff(t)
    w = np.zeros_like(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    didt = np.cumsum(d * w)
    # dI/dt is a staircase (impulses d2I/dt2 at the samples), so I is piecewise linear
    # with slope didt[j] on [t_j, t_j+1]: this is exactly the current whose kinks are
    # the SR impulses anemone convolves with.
    cur = np.concatenate([[0.0], np.cumsum(didt[:-1] * dt)])

    # zero the residual after turn-off
    big = np.where(np.abs(cur) > tail_tol)[0]
    i_last = int(big[-1]) + 1 if big.size else len(t) - 1
    i_last = min(i_last, len(t) - 1)
    t = t[:i_last + 1]
    cur = cur[:i_last + 1].copy()
    cur[-1] = 0.0

    # drop (near-)duplicate times
    keep = [0]
    for k in range(1, len(t)):
        if t[k] - t[keep[-1]] > 1e-12:
            keep.append(k)
    keep = np.array(keep)
    t = t[keep]
    cur = cur[keep]
    cur[-1] = 0.0

    # Ramer-Douglas-Peucker on (t, cur) with a vertical-distance criterion
    flag = np.zeros(len(t), dtype=bool)
    flag[0] = True
    flag[-1] = True
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        tt = t[a + 1:b]
        line = cur[a] + (cur[b] - cur[a]) * (tt - t[a]) / (t[b] - t[a])
        err = np.abs(cur[a + 1:b] - line)
        j = int(np.argmax(err))
        if err[j] > tol:
            m = a + 1 + j
            flag[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.column_stack([t[flag], cur[flag]])


def _gex_value(lines, key):
    """First value string of ``key=...`` in the GEX text lines (or None)."""
    pat = re.compile(r'^\s*%s\s*=\s*(.*)$' % re.escape(key))
    for ln in lines:
        m = pat.match(ln)
        if m:
            return m.group(1).strip()
    return None


def gex_apply_sr2(file_gex, file_sr2, file_out=None, tol=1e-5, disable_filters=True,
                  pad_half_cycle=True, showInfo=0):
    """Write a GEX in which the system response is folded into the waveform.

    The returned GEX can be used by every forward backend (GA-AEM, SimPEG, anemone)
    without the SR2 file.  anemone with ``file_gex`` + ``file_sr2`` remains the
    reference implementation.

    Parameters
    ----------
    file_gex : str
        Workbench GEX (its waveform may be a placeholder when the SR is used).
    file_sr2 : str
        System-response file; block ``n`` is used for ``[Channel n]``.
    file_out : str, optional
        Output path (default ``<gex stem>_sr.gex`` next to ``file_gex``).
    tol : float
        Waveform decimation tolerance on the unit current (default 1e-5).
    disable_filters : bool
        Set ``TiBLowPassFilter`` and ``RxCoilLPFilter*`` cut-offs to 1 GHz, since the
        SR already contains that filtering (default True).
    pad_half_cycle : bool
        Append a zero-current point one half repetition period after the first
        waveform point (GA-AEM wants a complete half cycle; harmless otherwise).
    showInfo : int
        Verbosity.

    Returns
    -------
    str
        Path of the written GEX.
    """
    if not os.path.isfile(file_gex):
        raise FileNotFoundError("gex_apply_sr2: %s does not exist" % file_gex)
    if not os.path.isfile(file_sr2):
        raise FileNotFoundError("gex_apply_sr2: %s does not exist" % file_sr2)
    if file_out is None:
        file_out = os.path.splitext(file_gex)[0] + '_sr.gex'

    with open(file_gex, 'r') as f:
        lines = []
        for ln in f:
            lines.append(ln.rstrip('\n'))
    blocks = _read_sr2_blocks(file_sr2)

    # --- channel -> moment, repetition frequency (from the [ChannelN] sections)
    chan_info = {}
    cur_ch = None
    for ln in lines:
        s = ln.strip()
        m = re.match(r'^\[Channel(\d+)\]$', s)
        if m:
            cur_ch = int(m.group(1))
            chan_info[cur_ch] = {}
        elif s.startswith('[') :
            cur_ch = None
        elif cur_ch is not None and '=' in s and not s.startswith('/'):
            k, v = s.split('=', 1)
            chan_info[cur_ch][k.strip()] = v.strip()

    waveforms = {}
    for ch in sorted(chan_info):
        if ch not in blocks:
            continue
        mom = chan_info[ch].get('TransmitterMoment', 'LM' if ch == 1 else 'HM').strip().upper()
        mom = 'HM' if mom.startswith('H') else 'LM'
        wf = sr2_effective_current(blocks[ch]['time'], blocks[ch]['dsrdt'], tol=tol)
        if pad_half_cycle and 'RepFreq' in chan_info[ch]:
            rep = float(chan_info[ch]['RepFreq'].split()[0])
            t_pad = wf[0, 0] + 0.5 / rep + 1e-6
            if rep > 0 and wf[-1, 0] < t_pad:
                wf = np.vstack([wf, [t_pad, 0.0]])
        waveforms[mom] = wf
        if showInfo > 0:
            print('gex_apply_sr2: channel %d (%s): %d waveform points' % (ch, mom, wf.shape[0]))
    if not waveforms:
        raise ValueError("gex_apply_sr2: no SR2 block matches a [ChannelN] of %s" % file_gex)

    # --- rewrite the GEX text
    out = []
    in_channel = False
    inserted = {}
    for ln in lines:
        s = ln.strip()
        if re.match(r'^Waveform(LM|HM)Point\d+\s*=', s):
            continue
        if s.startswith('['):
            in_channel = re.match(r'^\[Channel\d+\]$', s) is not None
        if in_channel and re.match(r'^SystemResponseConvolution\s*=', s):
            out.append('SystemResponseConvolution=0')
            continue
        if disable_filters and in_channel and re.match(r'^TiBLowPassFilter\s*=', s):
            order = s.split('=', 1)[1].split()[0]
            out.append('TiBLowPassFilter=%s %.2E' % (order, _FILTER_OFF_FCUT))
            continue
        if disable_filters and re.match(r'^RxCoilLPFilter\d*\s*=', s):
            key, val = s.split('=', 1)
            out.append('%s= %s %.2E' % (key.strip(), val.split()[0], _FILTER_OFF_FCUT))
            continue
        out.append(ln)
        for mom in waveforms:
            if mom not in inserted and s.startswith('/') and (mom + ' Waveform definition') in s:
                for k in range(waveforms[mom].shape[0]):
                    out.append('Waveform%sPoint%02d=   %.9E %.9E' %
                               (mom, k + 1, waveforms[mom][k, 0], waveforms[mom][k, 1]))
                inserted[mom] = True
    for mom in waveforms:
        if mom not in inserted:
            raise ValueError("gex_apply_sr2: no '%s Waveform definition' comment line found "
                             "in %s to place the waveform after" % (mom, file_gex))

    with open(file_out, 'w') as f:
        f.write('\n'.join(out) + '\n')
    if showInfo > 0:
        print('gex_apply_sr2: wrote %s' % file_out)
    return file_out
