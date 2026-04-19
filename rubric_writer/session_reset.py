"""Streamlit session cleanup for project switches, logout, and evaluate workflows."""

from __future__ import annotations

import copy

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


def reset_evaluate_tab_workflow_state(*, clear_infer_session: bool = True) -> None:
    """Reset Evaluate-tab workflow fields to idle defaults.

    After a **project switch**, infer session data is loaded from the DB first; pass
    ``clear_infer_session=False`` so ``infer_all_conversations`` / DP state are kept.
    """
    if clear_infer_session:
        st.session_state.infer_all_conversations = []
        st.session_state.infer_dp_messages = []
        st.session_state.infer_dp_conversation = None
        st.session_state.infer_decision_points = None
        st.session_state.infer_user_categorizations = {}
        st.session_state.infer_categorizations_complete = False
    else:
        _all = st.session_state.get("infer_all_conversations") or []
        if _all:
            _latest = _all[-1]
            _uc = _latest.get("user_categorizations") or {}
            st.session_state.infer_user_categorizations = copy.deepcopy(_uc)
            st.session_state.infer_categorizations_complete = bool(_uc)
            _cf = _latest.get("classification_feedback") or {}
            st.session_state.chat_classification_feedback = copy.deepcopy(_cf)
        else:
            st.session_state.infer_user_categorizations = {}
            st.session_state.infer_categorizations_complete = False
            st.session_state.chat_classification_feedback = {}

    st.session_state.infer_behavioral_result = None
    st.session_state.infer_expanded_dp = None
    st.session_state.infer_dp_dimension_confirmed = False
    st.session_state.infer_dp_user_mappings = {}
    st.session_state.infer_step6_generated_task = None
    st.session_state.infer_step6_writing_task = ""
    st.session_state.infer_step6_auto_gen_done = False
    st.session_state.infer_step6_custom_task_key_version = 0
    st.session_state.infer_step6_drafts = None
    st.session_state.infer_step6_draft_labels = None
    st.session_state.infer_step6_rubric_versions_used = None
    st.session_state.infer_step6_blind_ratings = None
    st.session_state.infer_step6_user_ranking = None
    st.session_state.infer_step6_user_dimension_checks = None
    st.session_state.infer_step6_llm_evaluations = None
    st.session_state.infer_step6_survey = None
    st.session_state.infer_step6_claim2_metrics = None
    st.session_state.infer_step6_claim3_metrics = None
    st.session_state.infer_pending_rubric = None

    if clear_infer_session:
        st.session_state.chat_criteria_llm_classification = None
        st.session_state.chat_criteria_user_classifications = {}
        st.session_state.chat_criteria_review_active = False
        st.session_state.chat_criteria_review_confirmed = False
        st.session_state.chat_classification_feedback = {}
        st.session_state.chat_criteria_hallucination_reasons = {}
        if "chat_criteria_importance_ranks" in st.session_state:
            del st.session_state.chat_criteria_importance_ranks

    st.session_state.build_rubric_a_idx = None
    st.session_state.build_rubric_b_idx = None
    st.session_state.build_edit_classification = None
    st.session_state.build_writing_task = ""
    st.session_state.build_draft_a = None
    st.session_state.build_draft_b = None
    st.session_state.build_draft_a_thinking = ""
    st.session_state.build_draft_b_thinking = ""
    st.session_state.build_blind_labels = None
    st.session_state.build_user_preference = None
    st.session_state.build_llm_judge_result = None
    st.session_state.build_llm_judge_thinking = ""
    st.session_state.build_self_report = {}
    st.session_state.build_self_report_saved = False

    st.session_state.grade_writing_task = ""
    st.session_state.grade_violated_dims = None
    st.session_state.grade_draft_good = None
    st.session_state.grade_draft_degraded = None
    st.session_state.grade_draft_good_thinking = ""
    st.session_state.grade_draft_degraded_thinking = ""
    st.session_state.grade_blind_labels = None
    st.session_state.grade_user_overall_pref = None
    st.session_state.grade_user_dim_ratings = {}
    st.session_state.grade_rubric_judge_result = None
    st.session_state.grade_rubric_judge_thinking = ""
    st.session_state.grade_generic_judge_result = None
    st.session_state.grade_generic_judge_thinking = ""
    st.session_state.grade_agreement_results = None
    st.session_state.grade_saved = False
