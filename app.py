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

# Layer 2: Ranking checkpoint state (alignment diagnostic)
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

# Cold-start preference description (still loaded into chat for context)
if 'infer_coldstart_text' not in st.session_state:
    st.session_state.infer_coldstart_text = ""
if 'infer_coldstart_saved' not in st.session_state:
    st.session_state.infer_coldstart_saved = False

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

_labels = ["💬 Chat", "📋 Evaluate: Survey", "⚖️ Evaluate: Comparison", "📁 View Rubric", "🔍 Compare Rubrics"]
_tabs = st.tabs(_labels)
tab1, tab_survey, tab_comparison, tab3, tab4 = _tabs

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

with tab_comparison:
    _safe_render("Evaluate: Comparison", render_comparison_tab)

with tab_survey:
    _safe_render("Evaluate: Survey", render_survey_tab)
