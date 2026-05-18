import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# init_chat_model(model=AI_MODEL, model_provider=AI_MODEL_PROVIDER)
AI_MODEL = "google_genai:gemini-3-flash-preview"

PIPELINE_DIR = PROJECT_ROOT / "pipeline"
PIPELINE_CONFIG = PIPELINE_DIR / "config.yaml"
DEFAULT_GENDER = "female"
