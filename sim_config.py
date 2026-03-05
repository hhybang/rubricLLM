"""Configuration dataclasses for the synthetic user simulation pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import json


# ── Supported model providers ────────────────────────────────────────────────
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"

# Default model IDs per provider
DEFAULT_MODELS = {
    PROVIDER_ANTHROPIC: {
        "primary": "claude-opus-4-6",
        "light": "claude-sonnet-4-6",
    },
    PROVIDER_OPENAI: {
        "primary": "gpt-5.2",
        "light": "gpt-5.2",
    },
    PROVIDER_GOOGLE: {
        "primary": "gemini-3.1-pro-preview",
        "light": "gemini-3.1-pro-preview",
    },
}


@dataclass
class Persona:
    """A synthetic user persona with stated and hidden preferences."""
    name: str                    # e.g., "Alex Chen"
    role: str                    # e.g., "Product Manager at B2B SaaS startup"
    writing_type: str            # e.g., "professional emails"
    project_name: str            # e.g., "Work Emails"
    core_preferences: str        # stated preferences (mentioned in cold-start)
    hidden_preferences: str      # latent preferences (emerge through feedback)
    dealbreakers: str            # hard rejections
    initial_task: str            # first writing request
    gold_draft: Optional[str] = None  # hand-authored ideal draft; auto-generated if absent

    @classmethod
    def from_dict(cls, d: dict) -> "Persona":
        # Filter to only known fields so extra/missing optional keys don't break
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SimConfig:
    """Configuration for the simulation pipeline."""
    # Scale
    num_users: int = 5
    num_iterations: int = 3           # rubric refinement iterations per user
    chat_turns_per_iteration: int = 3 # DEPRECATED: use max_chat_turns instead
    max_chat_turns: int = 10          # max drafts the LLM can produce before terminating
    satisfaction_threshold: float = 0.8  # threshold for both judges (0.0-1.0)

    # System models (always Anthropic — the RubricLLM system uses Claude)
    model_primary: str = "claude-opus-4-6"   # rubric inference, judging
    model_light: str = "claude-sonnet-4-6"   # chat responses, draft generation

    # Synthetic user model (can be any provider)
    user_provider: str = PROVIDER_ANTHROPIC   # "anthropic", "openai", or "google"
    user_model: str = "claude-sonnet-4-6"     # model ID for the synthetic user LLM
    user_temperature: float = 0.8             # higher temp = more variability

    # Features
    enable_probes: bool = False       # skip initially to save cost
    enable_log_changes: bool = True   # synthetic user directly edits rubric

    # Personas
    persona_file: str = "personas.json"

    # Supabase (loaded from .streamlit/secrets.toml or env)
    supabase_url: str = ""
    supabase_key: str = ""            # service role key preferred

    # Synthetic user ID prefix (to distinguish from real users in DB)
    synthetic_user_id_prefix: str = "sim_"

    def load_personas(self) -> list["Persona"]:
        """Load personas from JSON file."""
        import os
        path = os.path.join(os.path.dirname(__file__), self.persona_file)
        with open(path) as f:
            data = json.load(f)
        return [Persona.from_dict(p) for p in data]
