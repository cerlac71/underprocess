"""One-call project bootstrap for scripts and Streamlit entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SRC_ROOT.parent


def bootstrap_project(region: str | None = None) -> Path:
    """Configure import paths, region, matplotlib cache, and thread limits."""
    from config.threading_env import limit_cpu_threads

    for entry in (str(_PROJECT_ROOT), str(_SRC_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.chdir(_PROJECT_ROOT)
    if region:
        os.environ["SIH_REGION"] = region
    os.environ.setdefault("SIH_REGION", "uttar_pradesh")

    mpl_dir = _PROJECT_ROOT / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    limit_cpu_threads()
    return _PROJECT_ROOT
