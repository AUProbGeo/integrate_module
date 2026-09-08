"""INTEGRATE Workbench — a FastHTML front end for the ``integrate`` module.

Self-contained: nothing here is imported by the core ``integrate`` package,
and every heavy computation is delegated to ``integrate`` via
``frontend.services.integrate_api``. See ``frontend/FRONTEND.md`` for the
full plan and TODO checklist.
"""

__all__ = ["__version__"]
__version__ = "0.0.1"
