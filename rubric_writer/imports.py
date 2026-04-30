"""Shared imports for the Streamlit app (extracted from the original app.py)."""
import streamlit as st
import os
import anthropic
import html as html_module
from textwrap import dedent
import json
import re
import time
import uuid
import copy
from datetime import datetime
from pathlib import Path
from auth_supabase import (
    get_supabase_client, init_auth_state, register_user, login_user,
    send_otp, verify_otp,
    logout_user, get_current_user, is_authenticated,
    get_user_projects, create_project as db_create_project, delete_project,
    save_conversation, load_conversations as db_load_conversations, delete_conversation,
    load_conversation_by_id, save_rubric_history as db_save_rubric_history,
    load_rubric_history as db_load_rubric_history, save_project_data,
    load_project_data, get_schema_sql, delete_rubric_version
)
from prompts import (
    RUBRIC_COMPARE_DRAFTS_PROMPT,
    CHAT_build_system_prompt,
    RUBRIC_compare_to_coldstart_prompt,
    GRADING_generate_draft_from_rubric_prompt,
    GRADING_judge_per_dimension_prompt,
    GRADING_generate_degraded_draft_prompt,
    GRADING_rubric_judge_prompt,
    GRADING_rubric_judge_3draft_prompt,
    GRADING_generic_judge_prompt,
    GRADING_generate_draft_from_coldstart_prompt,
    RUBRIC_suggest_changes_from_feedback_prompt,
    RUBRIC_apply_suggestion_prompt,
    GRADING_generate_writing_task_prompt,
    GRADING_generate_draft_generic_prompt,
    GRADING_unified_eval_prompt,
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
    RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT,
    RUBRIC_extract_dps_user_prompt,
    RUBRIC_FINAL_INFER_SYSTEM_PROMPT,
    RUBRIC_final_infer_user_prompt,
    ALIGNMENT_diagnostic_suggest_and_apply_prompt,
    ALIGNMENT_generate_annotated_draft_prompt,
    ALIGNMENT_verify_suggested_rubric_prompt,
    PROBE_identify_uncertainty_prompt,
    PROBE_generate_variant_draft_prompt,
    PROBE_refine_criterion_prompt,
)

from scipy.stats import kendalltau
import random
import warnings
import logging
