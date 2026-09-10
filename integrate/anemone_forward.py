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


def gex_to_anemone_system(gex, showInfo=0):
    raise NotImplementedError  # Task 2


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
