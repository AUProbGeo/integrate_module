"""Service layer for the frontend.

- ``integrate_api``  — the ONLY module that imports ``integrate``.
- ``files``          — lightweight HDF5 structural scan/classify (vendored).
- ``workspace``      — workspace root + path confinement (vendored).
- ``figures``        — locked matplotlib -> PNG rendering.
- ``session``        — cookie-keyed per-session state.
- ``jobs`` / ``worker`` — child-process runner for long tasks.
"""
