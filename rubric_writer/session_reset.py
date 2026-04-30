"""Streamlit session cleanup for project switches, logout, and evaluate workflows."""

from __future__ import annotations

import streamlit as st


def clear_project_data_caches() -> None:
    """Drop per-project conversation and rubric history caches in session_state."""
    keys = list(st.session_state.keys())
    for k in keys:
        if not isinstance(k, str):
            continue
        if k.startswith("conversations_") or k.startswith("rubric_history_"):
            del st.session_state[k]


def clear_project_scoped_widget_keys() -> None:
    """Remove widget session keys namespaced with ``__proj__`` (see ``widget_keys.project_scoped_key``)."""
    keys = list(st.session_state.keys())
    for k in keys:
        if isinstance(k, str) and "__proj__" in k:
            del st.session_state[k]


def clear_cross_tab_project_state() -> None:
    """Reset UI state that must not leak across projects (compare, delete mode, etc.)."""
    st.session_state.comparison_result = None
    st.session_state.comparison_rubric_version = None
    st.session_state.rubric_comparison_results = None
    st.session_state.message_delete_mode = False
    st.session_state.messages_to_delete = set()
    st.session_state.current_analysis = ""
    st.session_state.current_rubric_assessment = None
