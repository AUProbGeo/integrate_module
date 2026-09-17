"""Per-session scratch state, server-side.

FastHTML gives each visitor a signed ``session`` cookie dict; we only keep a
stable id in it and hang the real (non-serialisable, larger) state off an
in-process dict here. Cleared on restart — fine for a single-user desktop
tool.
"""

from __future__ import annotations

import secrets
from typing import Any

_STORE: dict[str, dict[str, Any]] = {}


def state_for(session: dict) -> dict[str, Any]:
    """Return this visitor's mutable state dict, creating it on first use."""
    sid = session.get("wb_sid")
    if not sid or sid not in _STORE:
        sid = sid or secrets.token_hex(8)
        session["wb_sid"] = sid
        _STORE.setdefault(sid, {})
    return _STORE[sid]
