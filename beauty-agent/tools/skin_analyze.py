"""실제 SkinPipeline을 호출해 얼굴 이미지에서 raw 0~100 점수를 추출하는 tool.

Phase 1 설계:
- severity(severe/moderate/mild) 변환은 이 tool에서 하지 않는다.
- 호출자(LLM)에 raw 값을 그대로 노출 → 검색/설명 시점에 변환.
- 가중치가 큰 SkinPipeline 인스턴스는 gender별로 1회만 초기화하고 캐싱.
- Tool은 Command를 반환해 messages 외에 커스텀 state(skin_scores, top_concerns)를 함께 갱신한다.
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import yaml
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from config import DEFAULT_GENDER, PIPELINE_CONFIG, PIPELINE_DIR, PROJECT_ROOT

# pipeline.py 내부의 `from preprocess import ...` 가 동작하도록 pipeline/ 디렉토리를 sys.path에 추가
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _load_skin_pipeline_class():
    """pipeline/pipeline.py를 파일 경로로 직접 로드해 namespace 충돌을 피한다."""
    spec = importlib.util.spec_from_file_location(
        "_skin_pipeline_module", str(PIPELINE_DIR / "pipeline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SkinPipeline


def _runtime_config_path() -> Path:
    """config.yaml의 root_dir을 호스트의 실제 pipeline/ 경로로 덮어쓴 사본을 만든다."""
    with open(PIPELINE_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root_dir"] = str(PIPELINE_DIR)

    cache_dir = PROJECT_ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "pipeline_runtime_config.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return out


_pipeline_cache: dict = {}


def _get_pipeline(gender: str):
    gender = gender.lower()
    if gender not in _pipeline_cache:
        SkinPipeline = _load_skin_pipeline_class()
        _pipeline_cache[gender] = SkinPipeline(
            config_path=_runtime_config_path(), gender=gender
        )
    return _pipeline_cache[gender]


def _flatten(result: dict) -> dict:
    pigment = result.get("pigment") or {}
    wrinkle = result.get("wrinkle") or {}
    homogenity = result.get("homogenity") or {}
    cheek = result.get("cheek_sagging") or {}
    chin = result.get("chin_sagging") or {}
    return {
        "pigment_left":         pigment.get("left"),
        "pigment_right":        pigment.get("right"),
        "wrinkle_forehead":     wrinkle.get("forehead"),
        "wrinkle_right_eye":    wrinkle.get("right_eye"),
        "wrinkle_left_eye":     wrinkle.get("left_eye"),
        "wrinkle_nasolabial":   wrinkle.get("nasolabial"),
        "wrinkle_perioral":     wrinkle.get("perioral"),
        "wrinkle_right_vol":    wrinkle.get("right_vol"),
        "wrinkle_left_vol":     wrinkle.get("left_vol"),
        "homogenity_radiance":  homogenity.get("radiance"),
        "homogenity_texture":   homogenity.get("texture"),
        "cheek_sagging_total":  cheek.get("total"),
        "chin_sagging_total":   chin.get("total"),
    }


def _compute_top_concerns(raw_scores: dict, k: int = 3) -> list[dict]:
    """좌/우 등 대칭 점수를 부위별로 평균낸 뒤, 점수가 낮은(=상태가 심각한) 순서로 k개 추출."""
    def avg(*keys):
        vals = [raw_scores.get(key) for key in keys if raw_scores.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    candidates = [
        ("pigmentation",        "색소침착",        avg("pigment_left", "pigment_right")),
        ("forehead_wrinkle",    "이마주름",        avg("wrinkle_forehead")),
        ("eye_wrinkle",         "눈가주름",        avg("wrinkle_right_eye", "wrinkle_left_eye")),
        ("nasolabial_fold",     "팔자주름",        avg("wrinkle_nasolabial")),
        ("perioral_wrinkle",    "입가주름",        avg("wrinkle_perioral")),
        ("volume_wrinkle",      "볼륨 주름",       avg("wrinkle_right_vol", "wrinkle_left_vol")),
        ("homogenity_radiance", "피부 광채 균일도", avg("homogenity_radiance")),
        ("homogenity_texture",  "피부결 균일도",    avg("homogenity_texture")),
        ("cheek_sagging",       "볼 처짐",         avg("cheek_sagging_total")),
        ("chin_sagging",        "턱 처짐",         avg("chin_sagging_total")),
    ]
    regions = [
        {"region": en, "region_ko": ko, "score": round(score, 2)}
        for en, ko, score in candidates if score is not None
    ]
    return sorted(regions, key=lambda r: r["score"])[:k]


def _make_tool_message(content: dict, tool_call_id: str) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


@tool
def skin_analyze(
    image_path: str,
    gender: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """얼굴 이미지를 분석해 부위별 raw 점수(0~100, 낮을수록 심각)를 반환합니다.
    같은 이미지에 대해 한 번만 호출하세요. 중증도 변환은 별도 판단으로 수행합니다.

    피부 나이(age) 예측은 성별이 있어야만 신뢰할 수 있습니다.
    gender가 주어지지 않으면 age는 반환되지 않으며(age=null, age_note 동봉),
    다른 점수들은 fallback 성별(female)로 산출됩니다.

    Args:
        image_path: 얼굴 이미지 파일 경로 (절대 또는 작업 디렉토리 기준 상대 경로).
        gender: 'male' 또는 'female'. 미지정 시 age는 산출되지 않음.

    Returns:
        Command(update={...}) — state의 skin_scores/top_concerns 갱신 + ToolMessage 삽입.
    """
    gender_provided = bool(gender)
    effective_gender = (gender or DEFAULT_GENDER).lower()

    pipe = _get_pipeline(effective_gender)
    try:
        result = pipe.predict_single(image_path)
    except Exception as e:
        err = {"error": f"파이프라인 실행 실패: {e}", "image_path": image_path}
        return Command(update={"messages": [_make_tool_message(err, tool_call_id)]})

    if result is None:
        err = {"error": "얼굴을 검출하지 못했습니다.", "image_path": image_path}
        return Command(update={"messages": [_make_tool_message(err, tool_call_id)]})

    raw_scores = _flatten(result)
    top_concerns = _compute_top_concerns(raw_scores)

    skin_scores = {
        "image_path":    image_path,
        "gender_input":  gender,
        "age":           result.get("age") if gender_provided else None,
        "age_note":      None if gender_provided else "성별 미지정으로 피부 나이는 산출하지 않음. 사용자에게 성별을 확인 후 재호출 권장.",
        "valid_sagging": result.get("valid_sagging"),
        "raw_scores":    raw_scores,
    }

    # LLM이 ToolMessage에서 직접 볼 내용(top_concerns 포함)
    payload_for_llm = {**skin_scores, "top_concerns": top_concerns}

    return Command(update={
        "skin_scores":   skin_scores,
        "top_concerns":  top_concerns,
        "messages":      [_make_tool_message(payload_for_llm, tool_call_id)],
    })
