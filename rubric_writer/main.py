"""Backward compatibility for ``streamlit run rubric_writer/main.py``.

The canonical entry file is ``app.py`` at the repository root. That file contains
the full application; this module delegates to it so old commands keep working.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"
if not _APP.is_file():
    raise FileNotFoundError(f"Expected app entry at {_APP}")

runpy.run_path(str(_APP), run_name="__main__")
