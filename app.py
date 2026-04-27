"""AI-Rubric Writer — Streamlit entrypoint.

Run from the repository root::

    streamlit run app.py

Keeping the runnable script at the project root avoids import-only entrypoints
(``import rubric_writer.main``) that can behave differently with Streamlit’s
script runner and ``st.set_page_config`` / session lifecycle.
"""
from rubric_writer.imports import *
from rubric_writer import config  # noqa: F401 — initializes Anthropic client and model constants
from rubric_writer.persistence import load_rubric_history, get_active_rubric
from rubric_writer.session_reset import (
    clear_cross_tab_project_state,
    clear_project_data_caches,
    clear_project_scoped_widget_keys,
)

from rubric_writer.ui.tab_chat import render_chat_panel, render_chat_sidebar
from rubric_writer.ui.tab_view_rubric import render_view_rubric_tab
from rubric_writer.ui.tab_compare_rubrics import render_compare_rubrics_tab
from rubric_writer.ui.tab_infer import render_infer_tab
from rubric_writer.ui.tab_evaluate_build import render_evaluate_build_tab
from rubric_writer.ui.tab_evaluate_grade import render_evaluate_grade_tab
from rubric_writer.ui.tab_grading_dashboard import render_grading_dashboard_tab
from rubric_writer.ui.tab_survey import render_survey_tab
from rubric_writer.ui.tab_comparison import render_comparison_tab

st.set_page_config(
    page_title="AI Co-Writer",
    page_icon="✍️",
    layout="wide"
)

# (Scroll-to-top/bottom via components.v1 + parent.document removed — it could break
# Streamlit 1.50+ embedding and contribute to blank UI after reruns.)

# ========================
# Authentication Setup (Supabase)
# ========================
init_auth_state()

# Get Supabase client
supabase = get_supabase_client()

if supabase is None:
    st.error("Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY in your environment or Streamlit secrets.")
    st.markdown("### Setup Instructions")
    st.markdown("""
    1. Create a free account at [supabase.com](https://supabase.com)
    2. Create a new project
    3. Go to **Settings > API** to get your URL and anon key
    4. Add to your `.streamlit/secrets.toml`:
    ```toml
    SUPABASE_URL = "your-project-url"
    SUPABASE_KEY = "your-anon-key"
    ```
    5. Run the database schema (click below to copy):
    """)
    with st.expander("Database Schema SQL"):
        st.code(get_schema_sql(), language="sql")
    st.stop()

# Store supabase client in session state for use throughout the app
st.session_state.supabase = supabase

# Check if user is authenticated
if not is_authenticated():
    import hashlib as _auth_hashlib
    st.title("AI-Rubric Writer")
    st.markdown("Enter your name and email to get started.")

    # Check if we need to handle a legacy user (has old password)
    _legacy_email = st.session_state.get("_auth_legacy_email")

    if _legacy_email:
        # Legacy user migration: ask for old password one time
        st.info(f"Welcome back! We've simplified login — no more passwords. Please enter your existing password one last time for **{_legacy_email}** to migrate your account.")
        with st.form("legacy_form"):
            _legacy_pw = st.text_input("Your existing password", type="password")
            _legacy_submit = st.form_submit_button("Migrate & Continue", width="stretch")

            if _legacy_submit:
                if _legacy_pw:
                    success, message = login_user(supabase, _legacy_email, _legacy_pw)
                    if success:
                        st.session_state.pop("_auth_legacy_email", None)
                        st.session_state.supabase = get_supabase_client()
                        # Update password to auto-generated one for future logins
                        try:
                            _auto_pw = _auth_hashlib.sha256((_legacy_email + "_rubricllm_auto").encode()).hexdigest()[:24]
                            supabase_client = get_supabase_client()
                            supabase_client.auth.update_user({"password": _auto_pw})
                        except Exception:
                            pass  # Non-critical — will just ask for password again next time
                        st.rerun()
                    else:
                        st.error("Incorrect password. Please try again.")
                else:
                    st.warning("Please enter your password.")
        if st.button("Use a different email"):
            st.session_state.pop("_auth_legacy_email", None)
            st.rerun()
    else:
        with st.form("auth_form"):
            _auth_name = st.text_input("Name")
            _auth_email = st.text_input("Email")
            _auth_submit = st.form_submit_button("Continue", width="stretch")

            if _auth_submit:
                if not _auth_name or not _auth_email:
                    st.warning("Please enter both your name and email.")
                elif "@" not in _auth_email:
                    st.warning("Please enter a valid email address.")
                else:
                    _auth_email = _auth_email.strip().lower()
                    _auth_name = _auth_name.strip()
                    # Generate a deterministic password from the email (invisible to the user)
                    _auto_pw = _auth_hashlib.sha256((_auth_email + "_rubricllm_auto").encode()).hexdigest()[:24]
                    # Try to log in with auto password
                    success, message = login_user(supabase, _auth_email, _auto_pw)
                    if success:
                        st.session_state.supabase = get_supabase_client()
                        st.rerun()
                    else:
                        # Try to register (new user)
                        reg_success, reg_msg = register_user(supabase, _auth_email, _auto_pw, _auth_name)
                        if reg_success:
                            success, message = login_user(supabase, _auth_email, _auto_pw)
                            if success:
                                st.session_state.supabase = get_supabase_client()
                                st.rerun()
                        elif "already" in reg_msg.lower():
                            # Legacy user with old password — need one-time migration
                            st.session_state["_auth_legacy_email"] = _auth_email
                            st.rerun()
                        else:
                            st.error(reg_msg)

    st.stop()

# User is authenticated - get user info
current_user = get_current_user()
st.session_state.auth_username = current_user["id"]
st.session_state.auth_name = current_user["name"]
st.session_state.auth_email = current_user["email"]

# Ensure supabase client has the current session
st.session_state.supabase = get_supabase_client()

# Show logout button in sidebar
with st.sidebar:
    st.write(f'Welcome, **{current_user["name"]}**')
    if st.button("Logout", width="stretch"):
        logout_user(supabase)
        st.rerun()
    st.markdown("---")

# Inject CSS styles for diff highlighting
st.markdown("""
<style>
.diff-container {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
}
.diff-add {
    background-color: #d4edda;
    color: #155724;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.diff-del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 2px 4px;
    border-radius: 3px;
}
.diff-container p {
    margin: 0.5em 0;
}

/* Keep chat input at bottom */
div[data-testid="chatInputContainer"] {
    position: fixed !important;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: white;
    padding: 1rem;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Ensure chat input appears above all other content */
div[data-testid="chatInputContainer"] {
    z-index: 999 !important;
}

/* Add padding to main content area to prevent overlap */
main[data-testid="stMain"] {
    padding-bottom: 150px !important;
}

/* Also add padding to main block content */
.block-container {
    padding-bottom: 150px !important;
}

/* Ensure chat messages don't overlap the fixed input */
.stChatMessage {
    z-index: 1 !important;
}

/* Style for new criteria containers */
.criterion-container-new {
    border: 2px solid #FF9800 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(255, 152, 0, 0.1) !important;
}

.criterion-container-existing {
    border: 1px solid #2196F3 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(33, 150, 243, 0.05) !important;
}

/* Styles for rubric comparison diff highlighting */
.diff-wrap {
    font-family: monospace;
    line-height: 1.4;
}
.diff-wrap .add {
    background-color: #d4edda;
    color: #155724;
    padding: 1px 3px;
    border-radius: 2px;
    font-weight: bold;
}
.diff-wrap .del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 1px 3px;
    border-radius: 2px;
}
.diff-wrap p {
    margin: 0.5em 0;
}

/* Make tabs horizontally scrollable on small screens */
div[data-baseweb="tab-list"] {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 6px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* Prevent tabs from wrapping */
button[data-baseweb="tab"] {
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rubric' not in st.session_state:
    st.session_state.rubric = None

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = ""

if 'current_rubric_assessment' not in st.session_state:
    st.session_state.current_rubric_assessment = None

# Initialize current_project early so other functions can use it
if 'current_project' not in st.session_state:
    st.session_state.current_project = None  # Project name
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None  # Project UUID from Supabase
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = {"task_a": {}, "task_b": {}}

if 'selected_conversation' not in st.session_state:
    st.session_state.selected_conversation = None

if 'active_rubric_idx' not in st.session_state:
    hist = load_rubric_history()
    st.session_state.active_rubric_idx = len(hist) - 1 if hist else None
else:
    # Existing value may be stale (e.g. history grew or shrank between reruns).
    # Snap to last (highest version) if the index is out of range.
    _hist = load_rubric_history()
    _idx = st.session_state.active_rubric_idx
    if not _hist:
        st.session_state.active_rubric_idx = None
    elif _idx is None or _idx < 0 or _idx >= len(_hist):
        st.session_state.active_rubric_idx = len(_hist) - 1

# Comparison mode
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None  # Store regenerated response for comparison
if 'comparison_rubric_version' not in st.session_state:
    st.session_state.comparison_rubric_version = None  # Which rubric version was used for comparison

# Rubric comparison results (for Compare Rubrics tab)
if 'rubric_comparison_results' not in st.session_state:
    st.session_state.rubric_comparison_results = None

# Message deletion mode for removing unwanted messages from conversation
if 'message_delete_mode' not in st.session_state:
    st.session_state.message_delete_mode = False  # Whether delete mode is active
if 'messages_to_delete' not in st.session_state:
    st.session_state.messages_to_delete = set()  # Set of message indices to delete

# Uncertainty probe: every N-th draft, probe a rubric criterion the model is uncertain about
if 'probe_draft_counts' not in st.session_state:
    st.session_state.probe_draft_counts = {}  # Per-conversation draft counts: {conv_id: int}
if 'probe_pending' not in st.session_state:
    st.session_state.probe_pending = None  # Dict with probe variants when triggered
if 'probe_results' not in st.session_state:
    st.session_state.probe_results = []  # List of completed probe results

# Evaluation dashboard: grade evaluation and retest history
if 'grade_evaluation_history' not in st.session_state:
    st.session_state.grade_evaluation_history = []
if 'grade_retest_history' not in st.session_state:
    st.session_state.grade_retest_history = []
if 'diagnostic_retest_history' not in st.session_state:
    st.session_state.diagnostic_retest_history = []

# Layer 2: Ranking checkpoint state
if 'ranking_checkpoint_results' not in st.session_state:
    st.session_state.ranking_checkpoint_results = []  # List of completed checkpoint results
if 'ranking_checkpoint_pending' not in st.session_state:
    st.session_state.ranking_checkpoint_pending = None  # {step: 2|3, writing_task, drafts, shuffle_order}
if 'ranking_checkpoint_auto_triggered' not in st.session_state:
    st.session_state.ranking_checkpoint_auto_triggered = False
if 'alignment_check_done' not in st.session_state:
    st.session_state.alignment_check_done = False
if 'alignment_check_skipped' not in st.session_state:
    st.session_state.alignment_check_skipped = False


# Evaluate: Coverage tab state (9-step workflow)
# Evaluate: Infer tab state (11-step workflow)
if 'infer_coldstart_text' not in st.session_state:
    st.session_state.infer_coldstart_text = ""  # Step 1: User's cold-start preference description
if 'infer_coldstart_saved' not in st.session_state:
    st.session_state.infer_coldstart_saved = False  # Whether Step 1 submitted
if 'infer_user_categorizations' not in st.session_state:
    st.session_state.infer_user_categorizations = {}  # Steps 2-3: {"Criterion Name": "stated"|"real"|"hallucinated"}
if 'infer_categorizations_complete' not in st.session_state:
    st.session_state.infer_categorizations_complete = False  # Whether all criteria categorized
if 'infer_behavioral_result' not in st.session_state:
    st.session_state.infer_behavioral_result = None  # Step 5: LLM behavioral evidence (parsed JSON)
if 'infer_dp_conversation' not in st.session_state:
    st.session_state.infer_dp_conversation = None  # Step 4: Selected conversation for decision points
if 'infer_decision_points' not in st.session_state:
    st.session_state.infer_decision_points = None  # Step 4: Extracted decision points
if 'infer_all_conversations' not in st.session_state:
    st.session_state.infer_all_conversations = []  # List of {messages, decision_points, timestamp, rubric_version}
if 'infer_expanded_dp' not in st.session_state:
    st.session_state.infer_expanded_dp = None  # Step 5: Currently expanded decision point ID
if 'infer_dp_dimension_confirmed' not in st.session_state:
    st.session_state.infer_dp_dimension_confirmed = False  # Step 5: Whether user confirmed dimension mappings
if 'infer_dp_user_mappings' not in st.session_state:
    st.session_state.infer_dp_user_mappings = {}  # Step 5: User-confirmed dimension mappings {dp_id: {"criterion": name, "not_in_rubric_reason": str|None}}
if 'infer_step6_generated_task' not in st.session_state:
    st.session_state.infer_step6_generated_task = None
if 'infer_step6_writing_task' not in st.session_state:
    st.session_state.infer_step6_writing_task = ""
if 'infer_step6_auto_gen_done' not in st.session_state:
    st.session_state.infer_step6_auto_gen_done = False
if 'infer_step6_custom_task_key_version' not in st.session_state:
    st.session_state.infer_step6_custom_task_key_version = 0
if 'infer_step6_drafts' not in st.session_state:
    st.session_state.infer_step6_drafts = None  # {"r_star": str, "r1": str|None, "r0": str, "coldstart": str, "generic": str}
if 'infer_step6_draft_labels' not in st.session_state:
    st.session_state.infer_step6_draft_labels = None  # ordered list of source keys matching A/B/C/D/E shuffle
if 'infer_step6_rubric_versions_used' not in st.session_state:
    st.session_state.infer_step6_rubric_versions_used = None  # {"r_star": ver, "r1": ver|None, "r0": ver}
if 'infer_step6_blind_ratings' not in st.session_state:
    st.session_state.infer_step6_blind_ratings = None  # {"A": 1-5, ...}
if 'infer_step6_user_ranking' not in st.session_state:
    st.session_state.infer_step6_user_ranking = None  # ordered list most→least preferred
if 'infer_step6_user_dimension_checks' not in st.session_state:
    st.session_state.infer_step6_user_dimension_checks = None  # Only for R*, coldstart, generic
if 'infer_step6_llm_evaluations' not in st.session_state:
    st.session_state.infer_step6_llm_evaluations = None
if 'infer_step6_survey' not in st.session_state:
    st.session_state.infer_step6_survey = None  # {"accuracy": str, "rounds_needed": str}
if 'infer_step6_claim2_metrics' not in st.session_state:
    st.session_state.infer_step6_claim2_metrics = None
if 'infer_step6_claim3_metrics' not in st.session_state:
    st.session_state.infer_step6_claim3_metrics = None
if 'infer_pending_rubric' not in st.session_state:
    st.session_state.infer_pending_rubric = None  # Infer tab: Pending inferred rubric awaiting user review

# Chat tab: Criteria classification (Steps 1+2 integrated from Infer tab)
if 'chat_criteria_llm_classification' not in st.session_state:
    st.session_state.chat_criteria_llm_classification = None  # LLM comparison result dict
if 'chat_criteria_user_classifications' not in st.session_state:
    st.session_state.chat_criteria_user_classifications = {}  # {criterion_name: "stated"|"real"|"hallucinated"}
if 'chat_criteria_review_active' not in st.session_state:
    st.session_state.chat_criteria_review_active = False  # True while review UI is showing
if 'chat_criteria_review_confirmed' not in st.session_state:
    st.session_state.chat_criteria_review_confirmed = False  # True after user confirms
if 'chat_classification_feedback' not in st.session_state:
    st.session_state.chat_classification_feedback = {}  # Stored after classification confirm
if 'chat_criteria_hallucination_reasons' not in st.session_state:
    st.session_state.chat_criteria_hallucination_reasons = {}

# Evaluate: Build tab state (5-step workflow)
if 'build_rubric_a_idx' not in st.session_state:
    st.session_state.build_rubric_a_idx = None  # Step 1: Index of Rubric A in history
if 'build_rubric_b_idx' not in st.session_state:
    st.session_state.build_rubric_b_idx = None  # Step 1: Index of Rubric B in history
if 'build_edit_classification' not in st.session_state:
    st.session_state.build_edit_classification = None  # Step 2: Structured diff result
if 'build_writing_task' not in st.session_state:
    st.session_state.build_writing_task = ""  # Step 3: User's writing task description
if 'build_draft_a' not in st.session_state:
    st.session_state.build_draft_a = None  # Step 3: Draft from rubric A
if 'build_draft_b' not in st.session_state:
    st.session_state.build_draft_b = None  # Step 3: Draft from rubric B
if 'build_draft_a_thinking' not in st.session_state:
    st.session_state.build_draft_a_thinking = ""  # Step 3: Thinking from draft A
if 'build_draft_b_thinking' not in st.session_state:
    st.session_state.build_draft_b_thinking = ""  # Step 3: Thinking from draft B
if 'build_blind_labels' not in st.session_state:
    st.session_state.build_blind_labels = None  # Step 3: {"Draft X": "a", "Draft Y": "b"}
if 'build_user_preference' not in st.session_state:
    st.session_state.build_user_preference = None  # Step 3: User's blind preference + per-dimension ratings
if 'build_llm_judge_result' not in st.session_state:
    st.session_state.build_llm_judge_result = None  # Step 4: LLM judge per-dimension scores
if 'build_llm_judge_thinking' not in st.session_state:
    st.session_state.build_llm_judge_thinking = ""  # Step 4: LLM judge thinking
if 'build_self_report' not in st.session_state:
    st.session_state.build_self_report = {}  # Step 5: User's self-report responses
if 'build_self_report_saved' not in st.session_state:
    st.session_state.build_self_report_saved = False  # Step 5: Whether self-report submitted

# Evaluate: Grade tab state (5-step workflow)
if 'grade_writing_task' not in st.session_state:
    st.session_state.grade_writing_task = ""  # Step 1: User's writing task description
if 'grade_violated_dims' not in st.session_state:
    st.session_state.grade_violated_dims = None  # Step 1: List of dimension names selected for violation
if 'grade_draft_good' not in st.session_state:
    st.session_state.grade_draft_good = None  # Step 1: Draft following full rubric
if 'grade_draft_degraded' not in st.session_state:
    st.session_state.grade_draft_degraded = None  # Step 1: Draft with violated dimensions
if 'grade_draft_good_thinking' not in st.session_state:
    st.session_state.grade_draft_good_thinking = ""  # Step 1: Thinking from good draft
if 'grade_draft_degraded_thinking' not in st.session_state:
    st.session_state.grade_draft_degraded_thinking = ""  # Step 1: Thinking from degraded draft
if 'grade_blind_labels' not in st.session_state:
    st.session_state.grade_blind_labels = None  # Step 1: {"Draft X": "good"|"degraded", ...}
if 'grade_user_overall_pref' not in st.session_state:
    st.session_state.grade_user_overall_pref = None  # Step 2: User's overall preference
if 'grade_user_dim_ratings' not in st.session_state:
    st.session_state.grade_user_dim_ratings = {}  # Step 2: Per-dimension ratings for both drafts
if 'grade_rubric_judge_result' not in st.session_state:
    st.session_state.grade_rubric_judge_result = None  # Step 3: Rubric-grounded judge result
if 'grade_rubric_judge_thinking' not in st.session_state:
    st.session_state.grade_rubric_judge_thinking = ""  # Step 3: Rubric-grounded thinking
if 'grade_generic_judge_result' not in st.session_state:
    st.session_state.grade_generic_judge_result = None  # Step 3: Generic judge result
if 'grade_generic_judge_thinking' not in st.session_state:
    st.session_state.grade_generic_judge_thinking = ""  # Step 3: Generic thinking
if 'grade_agreement_results' not in st.session_state:
    st.session_state.grade_agreement_results = None  # Step 4: Computed correlations
if 'grade_saved' not in st.session_state:
    st.session_state.grade_saved = False  # Step 5: Whether results saved

# Alignment tab state
if 'alignment_selected_conversation' not in st.session_state:
    st.session_state.alignment_selected_conversation = None  # Selected conversation file
if 'alignment_selected_draft_idx' not in st.session_state:
    st.session_state.alignment_selected_draft_idx = None  # Index of selected draft in conversation
if 'alignment_draft_content' not in st.session_state:
    st.session_state.alignment_draft_content = None  # The actual draft text
if 'alignment_user_scores' not in st.session_state:
    st.session_state.alignment_user_scores = {}  # User's scores: {criterion_idx: score}
if 'alignment_llm_scores' not in st.session_state:
    st.session_state.alignment_llm_scores = None  # LLM's scores for comparison
if 'alignment_results' not in st.session_state:
    st.session_state.alignment_results = None  # Computed alignment metrics
if 'alignment_evidence_highlights' not in st.session_state:
    st.session_state.alignment_evidence_highlights = []  # Evidence highlights from LLM scoring

# Initialize rubric from active version if not set
if st.session_state.rubric is None and st.session_state.active_rubric_idx is not None:
    active_rubric_dict, _, _ = get_active_rubric()
    if active_rubric_dict:
        rubric_list = active_rubric_dict.get("rubric", [])
        st.session_state.rubric = rubric_list

# After a project switch, clear widget keys and caches *before* any tab renders.
# Otherwise the Chat tab runs first and can leave Streamlit's widget graph inconsistent
# with session_state (blank main area).
if st.session_state.pop("_pending_clear_widgets_after_project_switch", None):
    clear_project_data_caches()
    clear_project_scoped_widget_keys()
    clear_cross_tab_project_state()

# Title at the top
st.title("✍️ AI-Rubric Writer")
st.markdown("Collaborate with AI to improve your writing!")

# Create tabs (Evaluate: Build, Grade, Infer, and Grading hidden)
SHOW_BUILD_GRADE_TABS = False
SHOW_INFER_GRADING_TABS = False
_labels = ["💬 Chat", "📋 Evaluate: Survey"]
if SHOW_INFER_GRADING_TABS:
    _labels += ["🔎 Evaluate: Infer", "📊 Evaluate: Grading"]
if SHOW_BUILD_GRADE_TABS:
    _labels += ["🔨 Evaluate: Build", "📝 Evaluate: Grade"]
_labels += ["⚖️ Evaluate: Comparison", "📁 View Rubric", "🔍 Compare Rubrics"]
_tabs = st.tabs(_labels)
_idx = 0
tab1 = _tabs[_idx]; _idx += 1
tab_survey = _tabs[_idx]; _idx += 1
if SHOW_INFER_GRADING_TABS:
    tab_infer = _tabs[_idx]; _idx += 1
    tab_grading = _tabs[_idx]; _idx += 1
else:
    tab_infer = tab_grading = None
if SHOW_BUILD_GRADE_TABS:
    tab7 = _tabs[_idx]; _idx += 1
    tab8 = _tabs[_idx]; _idx += 1
else:
    tab7 = tab8 = None
tab_comparison = _tabs[_idx]; _idx += 1
tab3 = _tabs[_idx]; _idx += 1
tab4 = _tabs[_idx]; _idx += 1

def _safe_render(label: str, render_fn) -> None:
    """Surface exceptions in the UI instead of a blank main area."""
    try:
        render_fn()
    except Exception as _e:
        st.error(f"**{label}** crashed — details below.")
        st.exception(_e)


with tab1:
    _safe_render("Chat", render_chat_panel)

with st.sidebar:
    _safe_render("Sidebar", render_chat_sidebar)

with tab3:
    _safe_render("View Rubric", render_view_rubric_tab)

with tab4:
    _safe_render("Compare Rubrics", render_compare_rubrics_tab)
    # RQ2: Fresh-generation pairwise preference comparison
    from rubric_writer.metrics import render_pairwise_comparison
    with st.expander("Rubric A/B Comparison (Research)", expanded=False):
        render_pairwise_comparison()

if SHOW_INFER_GRADING_TABS and tab_infer is not None:
    with tab_infer:
        _safe_render("Evaluate: Infer", render_infer_tab)

if SHOW_BUILD_GRADE_TABS and tab7 is not None:
    with tab7:
        _safe_render("Evaluate: Build", render_evaluate_build_tab)

if SHOW_BUILD_GRADE_TABS and tab8 is not None:
    with tab8:
        _safe_render("Evaluate: Grade", render_evaluate_grade_tab)

with tab_comparison:
    _safe_render("Evaluate: Comparison", render_comparison_tab)

if SHOW_INFER_GRADING_TABS and tab_grading is not None:
    with tab_grading:
        _safe_render("Evaluate: Grading", render_grading_dashboard_tab)

with tab_survey:
    _safe_render("Evaluate: Survey", render_survey_tab)
