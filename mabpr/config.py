"""Central configuration for the mabpr agent layer."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")

def openai_api_key() -> str:
    """Return the OpenAI key from the environment, or fail loudly."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in the repo-root .env "
            "(it is loaded via mabpr.config) — never inline a key in code."
        )
    return key

FAST_MODEL = "gpt-4o"
REASONING_MODEL = "gpt-5.6-luna"

MAX_REASON_TOKENS = 1500
MAX_EXTRACT_TOKENS = 1200
