"""Synthetic-user evaluation pipeline (headless RubricLLM simulation).

Public API:
    ``SimConfig``, ``Persona``, provider constants, ``SyntheticUser``, and
    engine helpers in ``simulation.engine``. Run the CLI via ``python simulate.py``
    or ``python -m simulation``.
"""

from simulation.config import (
    DEFAULT_MODELS,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    Persona,
    SimConfig,
)
from simulation.synthetic_user import SyntheticUser

__all__ = [
    "DEFAULT_MODELS",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GOOGLE",
    "PROVIDER_OPENAI",
    "Persona",
    "SimConfig",
    "SyntheticUser",
]
