"""Experiment hyperparameters and model IDs."""

import os

# Models (override via env) — align with simulation/ grader_agreement_test IDs
GENERATION_MODEL = os.environ.get("EXPERIMENT_GENERATION_MODEL", "claude-sonnet-4-6")
GRADING_MODEL = os.environ.get("EXPERIMENT_GRADING_MODEL", "claude-sonnet-4-6")
CORRECTION_MODEL = os.environ.get("EXPERIMENT_CORRECTION_MODEL", "claude-sonnet-4-6")
GROUND_TRUTH_MODEL = os.environ.get("EXPERIMENT_GROUND_TRUTH_MODEL", "claude-opus-4-6")

GENERATION_TEMPERATURE = float(os.environ.get("EXPERIMENT_GENERATION_TEMPERATURE", "0.7"))
GRADING_TEMPERATURE = float(os.environ.get("EXPERIMENT_GRADING_TEMPERATURE", "0.0"))
CORRECTION_TEMPERATURE = float(os.environ.get("EXPERIMENT_CORRECTION_TEMPERATURE", "0.3"))
GROUND_TRUTH_TEMPERATURE = float(os.environ.get("EXPERIMENT_GROUND_TRUTH_TEMPERATURE", "0.0"))
BEST_OF_N_TEMPERATURE = float(os.environ.get("EXPERIMENT_BEST_OF_N_TEMPERATURE", "0.8"))

NUM_ROUNDS = int(os.environ.get("EXPERIMENT_NUM_ROUNDS", "5"))
BEST_OF_N = int(os.environ.get("EXPERIMENT_BEST_OF_N", "5"))
NUM_GRADING_RUNS = int(os.environ.get("EXPERIMENT_NUM_GRADING_RUNS", "3"))

MAX_CONCURRENT_CALLS = int(os.environ.get("EXPERIMENT_MAX_CONCURRENT", "5"))
RETRY_ATTEMPTS = int(os.environ.get("EXPERIMENT_RETRY_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS = float(os.environ.get("EXPERIMENT_RETRY_DELAY", "5"))

MAX_TOKENS_GENERATION = int(os.environ.get("EXPERIMENT_MAX_TOKENS_GEN", "8192"))
MAX_TOKENS_GRADING = int(os.environ.get("EXPERIMENT_MAX_TOKENS_GRADE", "8192"))

# Approximate USD per 1M tokens (edit when pricing changes; estimate only)
PRICE_PER_M_INPUT = float(os.environ.get("EXPERIMENT_PRICE_INPUT_PER_M", "3.0"))
PRICE_PER_M_OUTPUT = float(os.environ.get("EXPERIMENT_PRICE_OUTPUT_PER_M", "15.0"))
OPUS_MULTIPLIER = float(os.environ.get("EXPERIMENT_OPUS_PRICE_MULT", "5.0"))
