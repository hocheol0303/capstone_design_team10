"""Shared helpers for eval runners — env loading, paths, JSON dump."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve project root and load .env once
EVAL_DIR = Path(__file__).resolve().parent.parent
BEAUTY_AGENT_DIR = EVAL_DIR.parent
PROJECT_ROOT = BEAUTY_AGENT_DIR.parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BEAUTY_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(BEAUTY_AGENT_DIR))

try:
    import dotenv  # type: ignore
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(name: str) -> dict:
    with open(DATASETS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def save_result(name: str, data: dict) -> Path:
    """Save {name}.json (overwrite latest). Also keep timestamped copy for history."""
    out = RESULTS_DIR / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    ts_copy = RESULTS_DIR / f"{name}_{timestamp()}.json"
    with open(ts_copy, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return out
