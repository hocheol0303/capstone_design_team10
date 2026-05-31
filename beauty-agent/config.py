import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# init_chat_model(model=AI_MODEL, model_provider=AI_MODEL_PROVIDER)
AI_MODEL = "openai:gpt-4o"

# 대화형 노드(think/final_report)용 온도 — 친근하고 자연스러운 말투를 위해 약간만 높임.
# 의도분류·PubMed 검색어 생성·리랭킹 등 JSON을 뱉는 단계는 temperature=0(결정적)으로 유지한다.
AI_TEMPERATURE_CHAT = float(os.getenv("AI_TEMPERATURE_CHAT", "0.4"))

PIPELINE_DIR = PROJECT_ROOT / "pipeline"
PIPELINE_CONFIG = PIPELINE_DIR / "config.yaml"
DEFAULT_GENDER = "female"
