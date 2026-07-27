#!/usr/bin/env python3
"""Convenience entry-point for the self-hosted model API server.

This thin wrapper re-exports the reference implementation so operators can
run either::

    python server.py [options]
    python scripts/reference_self_hosted_server.py [options]

Both are identical; ``server.py`` exists for simpler CLI invocation.
"""

import sys
from pathlib import Path

_REF = Path(__file__).resolve().parents[0] / "scripts" / "reference_self_hosted_server.py"

if not _REF.exists():
    raise SystemExit(f"Reference server not found at {_REF}")

# Re-exec the reference server so all argument parsing, logging, and uvicorn
# bootstrapping stays in one place.
sys.argv = [str(_REF), *sys.argv[1:]]
exec(_REF.read_text())
