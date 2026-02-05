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
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple


def get_supabase_client() -> Optional[Client]:
    """Get Supabase client using credentials from secrets or environment.

    Caches the client in session state to avoid creating too many connections.
    """
    # Return cached client if available
    if '_supabase_client' in st.session_state and st.session_state._supabase_client is not None:
        client = st.session_state._supabase_client

        # Just update the session on the existing client if needed
        session = st.session_state.get('auth_session')
        if session:
            try:
                client.auth.set_session(session.access_token, session.refresh_token)
            except Exception:
                try:
                    refresh_response = client.auth.refresh_session(session.refresh_token)
                    if refresh_response and refresh_response.session:
                        st.session_state.auth_session = refresh_response.session
                except Exception:
                    st.session_state.auth_session = None
                    st.session_state.auth_user = None

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

        # If we have a stored session, restore it on the client
        session = st.session_state.get('auth_session')
        if session:
            try:
                # Try to set the session using the stored tokens
                client.auth.set_session(session.access_token, session.refresh_token)
            except Exception:
                # If that fails, try refreshing the session
                try:
                    refresh_response = client.auth.refresh_session(session.refresh_token)
                    # Save the new session back to session state
                    if refresh_response and refresh_response.session:
                        st.session_state.auth_session = refresh_response.session
                except Exception:
                    # Session is truly invalid, clear it
                    st.session_state.auth_session = None
                    st.session_state.auth_user = None

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
    """Register a new user"""
    try:
        # Sign up with Supabase Auth
        # The user profile in public.users is created automatically by a database trigger
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
            return True, "Registration successful! Please check your email to verify your account."
        else:
            return False, "Registration failed. Please try again."

    except Exception as e:
        error_msg = str(e)
        if "User already registered" in error_msg:
            return False, "An account with this email already exists."
        return False, f"Registration error: {error_msg}"


def login_user(supabase: Client, email: str, password: str) -> Tuple[bool, str]:
    """Log in an existing user"""
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
    except:
        pass
    st.session_state.auth_user = None
    st.session_state.auth_session = None
    # Clear cached client so a fresh one is created on next login
    if '_supabase_client' in st.session_state:
        del st.session_state._supabase_client

    # Clear all project and rubric related session state
    keys_to_clear = [
        'current_project_id', 'current_project', 'rubric', 'active_rubric_idx',
        'messages', 'survey_responses', 'rubric_comparison_results', 'editing_criteria',
        'rubric_chat_messages', 'rubric_chat_suggestion', 'rubric_chat_preview_draft'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Clear all rubric history caches (keys starting with rubric_history_)
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith('rubric_history_')]
    for key in keys_to_delete:
        del st.session_state[key]


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
                     rubric: Any, analysis: str = "") -> Optional[str]:
    """Save a conversation to the database"""
    try:
        response = supabase.table("conversations").insert({
            "project_id": project_id,
            "messages": json.dumps(messages),
            "rubric": json.dumps(rubric) if rubric else None,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        }).execute()

        if response.data:
            return response.data[0]["id"]
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
            conversations.append({
                "id": conv["id"],
                "timestamp": conv["created_at"],
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
        response = supabase.table("conversations").select("*").eq("id", conversation_id).single().execute()

        if response.data:
            conv = response.data
            return {
                "id": conv["id"],
                "timestamp": conv["created_at"],
                "messages": json.loads(conv["messages"]) if conv["messages"] else [],
                "rubric": json.loads(conv["rubric"]) if conv["rubric"] else None,
                "analysis": conv.get("analysis", "")
            }
        return None
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return None


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

        response = supabase.table("rubric_history").insert({
            "project_id": project_id,
            "version": next_version,
            "rubric_data": json.dumps(rubric_data),
            "created_at": datetime.now().isoformat()
        }).execute()

        if response.data:
            return response.data[0]["id"]
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
"""


def get_schema_sql() -> str:
    """Return the database schema SQL"""
    return DATABASE_SCHEMA
