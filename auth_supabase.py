"""
Supabase Authentication and Data Storage Module for RubricLLM

This module handles:
- User authentication (login, register, logout)
- User session management
- Project and data storage in Supabase

Setup Instructions:
1. Create a Supabase account at https://supabase.com
2. Create a new project
3. Go to Settings > API to get your URL and anon key
4. Add these to your Streamlit secrets or environment variables:
   - SUPABASE_URL
   - SUPABASE_KEY

Database Schema (run in Supabase SQL Editor):
-- See setup_database() function or SCHEMA.sql for the SQL commands
"""

import streamlit as st
from supabase import create_client, Client
import os
import json
import logging
import time as _auth_time
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

_auth_log = logging.getLogger(__name__)

# Refresh the session only if the access token expires within this many
# seconds. Supabase access tokens default to 1 hour (3600s); we refresh
# when there's less than 5 minutes left so a long in-flight request
# doesn't hit the boundary.
_TOKEN_REFRESH_BUFFER_SECONDS = 300


def _token_needs_refresh(session) -> bool:
    """Return True iff the session's access token is expired or expiring soon.

    We avoid refreshing unnecessarily: the previous behavior called
    `set_session` on every page render, which meant a long-running LLM
    stream could race with a token-refresh failure and log the user out.
    Only refresh when actually needed."""
    if session is None:
        return False
    expires_at = getattr(session, "expires_at", None)
    if expires_at is None:
        # Unknown expiry -- don't proactively refresh; trust the cached session.
        return False
    try:
        now = _auth_time.time()
        return (float(expires_at) - now) < _TOKEN_REFRESH_BUFFER_SECONDS
    except (TypeError, ValueError):
        return False


def _safe_refresh_session(client, session) -> None:
    """Attempt to refresh the session. On transient failure, keep the existing
    session rather than nulling it out -- a network hiccup during a long LLM
    stream shouldn't log the user out.

    Only truly-invalid-token errors should clear the session, and we detect
    those by inspecting the exception message. Any other failure is logged
    and ignored; the next call will retry."""
    refresh_token = getattr(session, "refresh_token", None)
    if not refresh_token:
        return
    try:
        refresh_response = client.auth.refresh_session(refresh_token)
        if refresh_response and getattr(refresh_response, "session", None):
            st.session_state.auth_session = refresh_response.session
            return
        # refresh returned no new session but didn't raise -- likely transient
        _auth_log.info("Supabase session refresh returned no new session; keeping current session.")
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in ("invalid refresh", "refresh_token_not_found",
                                   "token has expired", "refresh_token expired")):
            _auth_log.warning("Supabase refresh token is truly invalid; clearing session: %s", e)
            st.session_state.auth_session = None
            st.session_state.auth_user = None
        else:
            # Transient failure (network, rate limit, etc.) -- keep the
            # existing session so the user isn't logged out mid-request.
            _auth_log.warning("Supabase refresh failed transiently, keeping existing session: %s", e)


def get_supabase_client() -> Optional[Client]:
    """Get Supabase client using credentials from secrets or environment.

    Caches the client in session state to avoid creating too many connections.
    Only refreshes the session when the access token is actually expiring --
    previously this was called on every Streamlit rerun, which produced a
    race where a token refresh during a long LLM stream could null out the
    session and log the user out mid-generation."""
    # Return cached client if available
    if '_supabase_client' in st.session_state and st.session_state._supabase_client is not None:
        client = st.session_state._supabase_client

        session = st.session_state.get('auth_session')
        if session and _token_needs_refresh(session):
            _safe_refresh_session(client, session)

        return client

    # Create new client
    try:
        # Try Streamlit secrets first
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

        if not url or not key:
            return None

        client = create_client(url, key)

        # Cache the client
        st.session_state._supabase_client = client

        # If we have a stored session, restore it on the client. This path
        # only runs on first client creation (once per Streamlit session),
        # so the initial `set_session` is safe here.
        session = st.session_state.get('auth_session')
        if session:
            try:
                client.auth.set_session(session.access_token, session.refresh_token)
            except Exception as e:
                msg = str(e).lower()
                if any(s in msg for s in ("invalid refresh", "refresh_token_not_found",
                                           "token has expired", "refresh_token expired")):
                    # Try to refresh; if that also fails, clear the session.
                    _safe_refresh_session(client, session)
                else:
                    _auth_log.warning("set_session failed on fresh client, keeping session: %s", e)

        return client
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None


def init_auth_state():
    """Initialize authentication state in session"""
    if 'auth_user' not in st.session_state:
        st.session_state.auth_user = None
    if 'auth_session' not in st.session_state:
        st.session_state.auth_session = None


def register_user(supabase: Client, email: str, password: str, name: str) -> Tuple[bool, str]:
    """Register a new user (legacy password-based, kept for compatibility)"""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "name": name
                }
            }
        })

        if response.user:
            return True, "Registration successful! You can now log in."
        else:
            return False, "Registration failed. Please try again."

    except Exception as e:
        error_msg = str(e)
        if "User already registered" in error_msg:
            return False, "An account with this email already exists."
        return False, f"Registration error: {error_msg}"


def send_otp(supabase: Client, email: str, name: str = "") -> Tuple[bool, str]:
    """Send a one-time password (OTP) to the user's email.

    Works for both new and existing users — Supabase auto-creates the account
    if the email is not yet registered.
    """
    try:
        options = {}
        if name:
            options["data"] = {"name": name}
        supabase.auth.sign_in_with_otp({
            "email": email,
            "options": options,
        })
        return True, "A sign-in code has been sent to your email."
    except Exception as e:
        return False, f"Failed to send code: {e}"


def verify_otp(supabase: Client, email: str, token: str) -> Tuple[bool, str]:
    """Verify an OTP code and log the user in."""
    try:
        response = supabase.auth.verify_otp({
            "email": email,
            "token": token,
            "type": "email",
        })
        if response.user and response.session:
            st.session_state.auth_user = {
                "id": response.user.id,
                "email": response.user.email,
                "name": response.user.user_metadata.get("name", email.split("@")[0]),
            }
            st.session_state.auth_session = response.session
            return True, "Login successful!"
        else:
            return False, "Verification failed. Please try again."
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "expired" in error_msg.lower():
            return False, "Invalid or expired code. Please request a new one."
        return False, f"Verification error: {error_msg}"


def login_user(supabase: Client, email: str, password: str) -> Tuple[bool, str]:
    """Log in an existing user (legacy password-based, kept for compatibility)"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if response.user and response.session:
            st.session_state.auth_user = {
                "id": response.user.id,
                "email": response.user.email,
                "name": response.user.user_metadata.get("name", email.split("@")[0])
            }
            st.session_state.auth_session = response.session
            return True, "Login successful!"
        else:
            return False, "Invalid credentials."

    except Exception as e:
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            return False, "Invalid email or password."
        return False, f"Login error: {error_msg}"


def logout_user(supabase: Client):
    """Log out the current user"""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.auth_user = None
    st.session_state.auth_session = None

    # Login view does not run the main app's session init; stale Streamlit widget keys
    # from the previous session must be cleared or remounting the app can blank the UI.
    from rubric_writer.session_reset import clear_project_data_caches, clear_project_scoped_widget_keys

    clear_project_data_caches()
    clear_project_scoped_widget_keys()

    for _k in ("auth_username", "auth_name", "auth_email"):
        st.session_state.pop(_k, None)

    # Clear cached client so a fresh one is created on next login
    if "_supabase_client" in st.session_state:
        del st.session_state._supabase_client

    # Clear all project and rubric related session state
    keys_to_clear = [
        "current_project_id",
        "current_project",
        "rubric",
        "active_rubric_idx",
        "messages",
        "survey_responses",
        "rubric_comparison_results",
        "editing_criteria",
        "rubric_chat_messages",
        "rubric_chat_suggestion",
        "rubric_chat_preview_draft",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def get_current_user() -> Optional[Dict]:
    """Get the current logged-in user"""
    return st.session_state.get('auth_user')


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('auth_user') is not None


# ========================
# Project Storage Functions
# ========================

def get_user_projects(supabase: Client, user_id: str) -> List[Dict]:
    """Get all projects for a user"""
    try:
        response = supabase.table("projects").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data or []
    except Exception as e:
        st.error(f"Error loading projects: {e}")
        return []


def create_project(supabase: Client, user_id: str, project_name: str) -> Tuple[bool, str, Optional[str]]:
    """Create a new project for a user. Returns (success, message, project_id)"""
    try:
        # Check if project name already exists for this user
        existing = supabase.table("projects").select("id").eq("user_id", user_id).eq("name", project_name).execute()
        if existing.data:
            return False, "A project with this name already exists.", None

        response = supabase.table("projects").insert({
            "user_id": user_id,
            "name": project_name,
            "created_at": datetime.now().isoformat()
        }).execute()

        if response.data:
            return True, f"Project '{project_name}' created successfully!", response.data[0]["id"]
        return False, "Failed to create project.", None

    except Exception as e:
        return False, f"Error creating project: {e}", None


def delete_project(supabase: Client, user_id: str, project_id: str) -> Tuple[bool, str]:
    """Delete a project and all its data"""
    try:
        # Delete associated data first (conversations, rubrics)
        supabase.table("conversations").delete().eq("project_id", project_id).execute()
        supabase.table("rubric_history").delete().eq("project_id", project_id).execute()

        # Delete the project
        supabase.table("projects").delete().eq("id", project_id).eq("user_id", user_id).execute()

        return True, "Project deleted successfully."
    except Exception as e:
        return False, f"Error deleting project: {e}"


# ========================
# Conversation Storage
# ========================

def save_conversation(supabase: Client, project_id: str, messages: List[Dict],
                     rubric: Any, analysis: str = "",
                     conversation_id: Optional[str] = None) -> Optional[str]:
    """Save a conversation to the database.

    If conversation_id is provided, updates the existing row.
    Otherwise, inserts a new row.
    """
    try:
        # Same sanitizer as save_rubric_history: strip display-only `_diff`
        # keys and coerce any stray set() to a list so json.dumps doesn't
        # crash on sets that leaked in from the display layer.
        def _sanitize(x):
            if isinstance(x, dict):
                return {k: _sanitize(v) for k, v in x.items() if k != "_diff"}
            if isinstance(x, list):
                return [_sanitize(v) for v in x]
            if isinstance(x, set):
                return sorted(x)
            return x
        data = {
            "project_id": project_id,
            "messages": json.dumps(_sanitize(messages)),
            "rubric": json.dumps(_sanitize(rubric)) if rubric else None,
            "analysis": analysis,
        }
        if conversation_id:
            # Delete + re-insert (no UPDATE RLS policy exists, so update silently fails)
            try:
                supabase.table("conversations").delete().eq("id", conversation_id).execute()
            except Exception:
                pass

        # Insert conversation (new or replacement)
        data["created_at"] = datetime.now().isoformat()
        if conversation_id:
            data["id"] = conversation_id  # Preserve the original ID
        response = supabase.table("conversations").insert(data).execute()
        if response.data:
            new_id = response.data[0]["id"]
            return new_id
        return None
    except Exception as e:
        st.error(f"Error saving conversation: {e}")
        return None


def load_conversations(supabase: Client, project_id: str) -> List[Dict]:
    """Load all conversations for a project"""
    try:
        response = supabase.table("conversations").select("*").eq("project_id", project_id).order("created_at", desc=True).execute()

        conversations = []
        for conv in response.data or []:
            # Clean Supabase timestamp: parse and re-format for display
            raw_ts = conv["created_at"] or ""
            try:
                from dateutil.parser import parse as _parse_dt
                clean_ts = _parse_dt(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                # Fallback: strip fractional seconds and timezone manually
                try:
                    _ts_clean = raw_ts.split(".")[0] if "." in raw_ts else raw_ts.split("+")[0]
                    from datetime import datetime as _dt
                    clean_ts = _dt.fromisoformat(_ts_clean).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    clean_ts = raw_ts
            conversations.append({
                "id": conv["id"],
                "timestamp": clean_ts,
                "messages": json.loads(conv["messages"]) if conv["messages"] else [],
                "rubric": json.loads(conv["rubric"]) if conv["rubric"] else None,
                "analysis": conv.get("analysis", ""),
                "messages_count": len(json.loads(conv["messages"])) if conv["messages"] else 0
            })
        return conversations
    except Exception as e:
        st.error(f"Error loading conversations: {e}")
        return []


def load_conversation_by_id(supabase: Client, conversation_id: str) -> Optional[Dict]:
    """Load a specific conversation by ID"""
    try:
        response = supabase.table("conversations").select("*").eq("id", conversation_id).maybe_single().execute()

        if response.data:
            conv = response.data
            raw_ts = conv["created_at"] or ""
            try:
                from datetime import datetime as _dt
                clean_ts = _dt.fromisoformat(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                clean_ts = raw_ts
            return {
                "id": conv["id"],
                "timestamp": clean_ts,
                "messages": json.loads(conv["messages"]) if conv["messages"] else [],
                "rubric": json.loads(conv["rubric"]) if conv["rubric"] else None,
                "analysis": conv.get("analysis", "")
            }
        return None
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return None


def delete_conversation(supabase: Client, conversation_id: str) -> bool:
    """Delete a conversation by ID. Returns True on success."""
    try:
        response = supabase.table("conversations").delete().eq("id", conversation_id).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        return False


# ========================
# Draft grades (background rubric grading)
# ========================

def insert_draft_grade(
    supabase: Client,
    *,
    conversation_id: str,
    message_id: Optional[str],
    draft_index: int,
    draft_text: str,
    rubric_version: int,
    grades_json: Dict[str, Any],
    model_used: str,
    latency_ms: Optional[int],
    trigger: str,
    drift_json: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist one grading result. Safe to call from a background thread (no st.*).

    AUTHORITATIVE DEDUP GUARD: if a row for this (conversation_id, message_id)
    already exists, skip the insert entirely. The process-local in-flight set
    and the pre-grade DB check can both lose to races across threads /
    processes; this final check before the insert closes those windows.
    """
    try:
        # Last-chance dedup: has this draft already been graded?
        if message_id:
            try:
                existing = (
                    supabase.table("draft_grades")
                    .select("id")
                    .eq("conversation_id", conversation_id)
                    .eq("message_id", message_id)
                    .limit(1)
                    .execute()
                )
                if existing.data:
                    import logging
                    logging.getLogger(__name__).info(
                        "insert_draft_grade: message_id=%s already has a row; "
                        "skipping duplicate insert.", message_id,
                    )
                    return False
            except Exception:
                # If the pre-check fails, proceed with insert; duplicates are
                # still handled by merge_draft_grades_into_messages dedup.
                pass

        row: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "draft_index": draft_index,
            "draft_text": draft_text,
            "rubric_version": rubric_version,
            "grades_json": grades_json,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "trigger": trigger,
        }
        if drift_json is not None:
            row["drift_json"] = drift_json
        resp = supabase.table("draft_grades").insert(row).execute()
        import logging as _lg
        n_rows = len(resp.data) if getattr(resp, "data", None) else 0
        _lg.getLogger(__name__).info(
            "[insert_draft_grade] INSERT for mid=%s draft_idx=%s → response.data has %s rows",
            message_id, draft_index, n_rows,
        )
        # RLS can silently drop an insert (API returns 201 + empty data) when
        # the with_check clause fails. That was happening in Session 1:
        # every draft looked like it graded successfully, but zero rows
        # landed in draft_grades. Treat empty response.data as a failure so
        # the caller knows the grade was NOT persisted, instead of silently
        # losing it.
        if n_rows == 0:
            _lg.getLogger(__name__).error(
                "[insert_draft_grade] INSERT returned 0 rows for mid=%s "
                "draft_idx=%s conv=%s — likely RLS with_check rejection. "
                "Check that the grading thread's auth token is being "
                "propagated to the PostgREST client (see draft_grading.py).",
                message_id, draft_index, conversation_id,
            )
            return False
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("insert_draft_grade failed: %s", e)
        return False


def fetch_draft_grades_for_conversation(supabase: Client, conversation_id: str) -> List[Dict[str, Any]]:
    """Return all draft_grades rows for a conversation, ordered by draft_index."""
    try:
        response = (
            supabase.table("draft_grades")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("draft_index")
            .execute()
        )
        return response.data or []
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("fetch_draft_grades_for_conversation failed: %s", e)
        return []


def next_draft_grade_index(supabase: Client, conversation_id: str) -> int:
    """Next sequential draft_index for this conversation (1-based)."""
    try:
        response = (
            supabase.table("draft_grades")
            .select("draft_index")
            .eq("conversation_id", conversation_id)
            .order("draft_index", desc=True)
            .limit(1)
            .execute()
        )
        import logging as _lg
        data = response.data if getattr(response, "data", None) else []
        _lg.getLogger(__name__).info(
            "[next_draft_grade_index] conv_id=%s → %d row(s); max_draft_index=%s",
            conversation_id, len(data),
            data[0].get("draft_index") if data else None,
        )
        if response.data:
            return int(response.data[0]["draft_index"]) + 1
    except Exception as e:
        import logging as _lg
        _lg.getLogger(__name__).warning("[next_draft_grade_index] fetch failed: %s", e)
    return 1


# ========================
# Rubric History Storage
# ========================

def save_rubric_history(supabase: Client, project_id: str, rubric_data: Dict) -> Optional[str]:
    """Save a rubric version to history"""
    try:
        # Get next version number
        existing = supabase.table("rubric_history").select("version").eq("project_id", project_id).order("version", desc=True).limit(1).execute()
        next_version = 1
        if existing.data:
            next_version = existing.data[0]["version"] + 1

        # Sanitize the rubric for JSON serialization. The display layer used
        # to write a `_diff` key on criteria that contained set() values;
        # json.dumps refuses sets. Strip `_diff` on its way out, and convert
        # any stray set() anywhere in the tree to a list as a safety net.
        def _sanitize(x):
            if isinstance(x, dict):
                return {k: _sanitize(v) for k, v in x.items() if k != "_diff"}
            if isinstance(x, list):
                return [_sanitize(v) for v in x]
            if isinstance(x, set):
                return sorted(x)
            return x
        _clean = _sanitize(rubric_data)

        response = supabase.table("rubric_history").insert({
            "project_id": project_id,
            "version": next_version,
            "rubric_data": json.dumps(_clean),
            "created_at": datetime.now().isoformat()
        }).execute()

        if response.data:
            return {"id": response.data[0]["id"], "version": next_version}
        return None
    except Exception as e:
        st.error(f"Error saving rubric: {e}")
        return None


def load_rubric_history(supabase: Client, project_id: str) -> List[Dict]:
    """Load rubric history for a project"""
    try:
        response = supabase.table("rubric_history").select("*").eq("project_id", project_id).order("version").execute()

        history = []
        for item in response.data or []:
            rubric_data = json.loads(item["rubric_data"]) if item["rubric_data"] else {}
            rubric_data["version"] = item["version"]
            rubric_data["id"] = item["id"]
            history.append(rubric_data)
        return history
    except Exception as e:
        st.error(f"Error loading rubric history: {e}")
        return []


def delete_rubric_version(supabase: Client, rubric_id: str) -> bool:
    """Delete a specific rubric version by its ID"""
    try:
        supabase.table("rubric_history").delete().eq("id", rubric_id).execute()

        # Verify deletion worked
        verify = supabase.table("rubric_history").select("id").eq("id", rubric_id).execute()
        return len(verify.data) == 0
    except Exception as e:
        st.error(f"Error deleting rubric version: {e}")
        return False


# ========================
# Generic Data Storage
# ========================

def save_project_data(supabase: Client, project_id: str, data_type: str, data: Any) -> bool:
    """Save generic project data (evaluations, surveys, etc.)"""
    try:
        # Check if data of this type already exists
        existing = supabase.table("project_data").select("id", "data").eq("project_id", project_id).eq("data_type", data_type).execute()

        if existing.data:
            # Append to existing data
            existing_data = json.loads(existing.data[0]["data"]) if existing.data[0]["data"] else []
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]

            supabase.table("project_data").update({
                "data": json.dumps(existing_data),
                "updated_at": datetime.now().isoformat()
            }).eq("id", existing.data[0]["id"]).execute()
        else:
            # Create new entry
            supabase.table("project_data").insert({
                "project_id": project_id,
                "data_type": data_type,
                "data": json.dumps([data]),
                "created_at": datetime.now().isoformat()
            }).execute()

        return True
    except Exception as e:
        st.error(f"Error saving project data: {e}")
        return False


def load_project_data(supabase: Client, project_id: str, data_type: str) -> List[Any]:
    """Load generic project data"""
    try:
        response = supabase.table("project_data").select("data").eq("project_id", project_id).eq("data_type", data_type).execute()

        if response.data and response.data[0]["data"]:
            return json.loads(response.data[0]["data"])
        return []
    except Exception as e:
        st.error(f"Error loading project data: {e}")
        return []


# ========================
# Database Schema Setup
# ========================

DATABASE_SCHEMA = """
-- Run this SQL in your Supabase SQL Editor to set up the database

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Projects table
CREATE TABLE IF NOT EXISTS public.projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- Conversations table
CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    messages JSONB,
    rubric JSONB,
    analysis TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Rubric history table
CREATE TABLE IF NOT EXISTS public.rubric_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    rubric_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, version)
);

-- Generic project data table (for evaluations, surveys, etc.)
CREATE TABLE IF NOT EXISTS public.project_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    data_type TEXT NOT NULL,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    UNIQUE(project_id, data_type)
);

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rubric_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_data ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own profile" ON public.users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.users
    FOR UPDATE USING (auth.uid() = id);

-- Projects policies
CREATE POLICY "Users can view own projects" ON public.projects
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own projects" ON public.projects
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own projects" ON public.projects
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own projects" ON public.projects
    FOR DELETE USING (auth.uid() = user_id);

-- Conversations policies
CREATE POLICY "Users can view own conversations" ON public.conversations
    FOR SELECT USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can create conversations in own projects" ON public.conversations
    FOR INSERT WITH CHECK (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can update own conversations" ON public.conversations
    FOR UPDATE USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can delete own conversations" ON public.conversations
    FOR DELETE USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

-- Rubric history policies
CREATE POLICY "Users can view own rubric history" ON public.rubric_history
    FOR SELECT USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can create rubric history in own projects" ON public.rubric_history
    FOR INSERT WITH CHECK (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

-- Project data policies
CREATE POLICY "Users can view own project data" ON public.project_data
    FOR SELECT USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can manage own project data" ON public.project_data
    FOR ALL USING (
        project_id IN (SELECT id FROM public.projects WHERE user_id = auth.uid())
    );

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON public.projects(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON public.conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_rubric_history_project_id ON public.rubric_history(project_id);
CREATE INDEX IF NOT EXISTS idx_project_data_project_id ON public.project_data(project_id);

-- Per-draft rubric grades (background Sonnet grading)
CREATE TABLE IF NOT EXISTS public.draft_grades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    message_id TEXT,
    draft_index INTEGER NOT NULL,
    draft_text TEXT NOT NULL,
    rubric_version INTEGER NOT NULL,
    grades_json JSONB NOT NULL,
    graded_at TIMESTAMPTZ DEFAULT NOW(),
    model_used TEXT DEFAULT 'claude-sonnet-4-6',
    latency_ms INTEGER,
    trigger TEXT NOT NULL,
    drift_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_draft_grades_conversation_id ON public.draft_grades(conversation_id);
CREATE INDEX IF NOT EXISTS idx_draft_grades_message_id ON public.draft_grades(message_id);

ALTER TABLE public.draft_grades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view draft grades for own conversations" ON public.draft_grades
    FOR SELECT USING (
        conversation_id IN (
            SELECT id FROM public.conversations WHERE project_id IN (
                SELECT id FROM public.projects WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users can insert draft grades for own conversations" ON public.draft_grades
    FOR INSERT WITH CHECK (
        conversation_id IN (
            SELECT id FROM public.conversations WHERE project_id IN (
                SELECT id FROM public.projects WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users can update draft grades for own conversations" ON public.draft_grades
    FOR UPDATE USING (
        conversation_id IN (
            SELECT id FROM public.conversations WHERE project_id IN (
                SELECT id FROM public.projects WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users can delete draft grades for own conversations" ON public.draft_grades
    FOR DELETE USING (
        conversation_id IN (
            SELECT id FROM public.conversations WHERE project_id IN (
                SELECT id FROM public.projects WHERE user_id = auth.uid()
            )
        )
    );
"""


def get_schema_sql() -> str:
    """Return the database schema SQL"""
    return DATABASE_SCHEMA
