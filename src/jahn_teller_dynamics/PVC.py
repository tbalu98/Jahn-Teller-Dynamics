"""
Backward-compatible alias for :mod:`jahn_teller_dynamics.jtd_run`.

Prefer the ``jtd_run`` console command or ``python -m jahn_teller_dynamics.jtd_run``.
The ``PVC`` entry point is kept so existing job scripts keep working.
"""

from jahn_teller_dynamics.jtd_run import *  # noqa: F403
