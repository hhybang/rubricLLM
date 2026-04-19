"""Project-scoped Streamlit widget keys.

Streamlit 1.50+ ties widget identity primarily to ``key``. Reusing the same key
after ``options``/context change (e.g. new project) can leave stale internal
state and a blank main area. Suffixing with ``current_project_id`` makes each
project a fresh widget namespace.
"""

from __future__ import annotations

import streamlit as st


def project_scoped_key(base: str) -> str:
    pid = st.session_state.get("current_project_id")
    if not pid:
        return f"{base}__proj__none"
    return f"{base}__proj__{pid}"
