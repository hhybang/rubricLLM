"""Model configuration and Anthropic client (extracted from app.py)."""
from rubric_writer.imports import st, os, anthropic, warnings, logging

# Suppress Streamlit deprecation warnings from terminal
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*Please replace.*")
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)
st_logger = logging.getLogger("streamlit")
st_logger.setLevel(logging.ERROR)

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_PRIMARY = "claude-opus-4-7"
MODEL_LIGHT = "claude-sonnet-4-6"
PROBE_FALLBACK_INTERVAL = 3

# Set the API key - check Streamlit secrets first, then environment variable
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv('ANTHROPIC_API_KEY')

if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

client = anthropic.Anthropic()
