"""Authentication, authorization and rate limiting for the HTTP API.

Everything in this package is plain Python (no torch, no database) so the
``backend.keys`` CLI and torch-free tests can import it.  ``policy.py`` is the
single place routes are classified; ``install.py`` wires the middlewares into
the FastAPI application.
"""

from .install import charge, get_runtime, install_security
from .principal import ANONYMOUS, Principal, current_principal, get_principal, principal_scope, require_admin

__all__ = [
    "ANONYMOUS",
    "Principal",
    "charge",
    "current_principal",
    "get_principal",
    "get_runtime",
    "install_security",
    "principal_scope",
    "require_admin",
]
