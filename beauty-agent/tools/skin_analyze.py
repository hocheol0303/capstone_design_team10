"""실제 SkinPipeline을 호출해 얼굴 이미지에서 raw 0~100 점수를 추출하는 tool.

Phase 1 설계:
- severity(severe/moderate/mild) 변환은 이 tool에서 하지 않는다.
- 호출자(LLM)에 raw 값을 그대로 노출 → 검색/설명 시점에 변환.
- 가중치가 큰 SkinPipeline 인스턴스는 gender별로 1회만 초기화하고 캐싱.
"""
import importlib.util
import sys
from pathlib import Path

import yaml
from langchain_core.tools import tool

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


@tool
def skin_analyze(image_path: str, gender: str | None = None) -> dict:
    """얼굴 이미지를 분석해 부위별 raw 점수(0~100, 낮을수록 심각)를 반환합니다.
    같은 이미지에 대해 한 번만 호출하세요. 중증도 변환은 별도 판단으로 수행합니다.

    피부 나이(age) 예측은 성별이 있어야만 신뢰할 수 있습니다.
    gender가 주어지지 않으면 age는 반환되지 않으며(age=null, age_note 동봉),
    다른 점수들은 fallback 성별(female)로 산출됩니다.

    Args:
        image_path: 얼굴 이미지 파일 경로 (절대 또는 작업 디렉토리 기준 상대 경로).
        gender: 'male' 또는 'female'. 미지정 시 age는 산출되지 않음.

    Returns:
        - image_path, gender_input, age, age_note, valid_sagging
        - raw_scores: 부위별 0~100 dict
        - error: 얼굴 미검출 또는 파이프라인 실패 시
    """
    gender_provided = bool(gender)
    effective_gender = (gender or DEFAULT_GENDER).lower()

    pipe = _get_pipeline(effective_gender)
    try:
        result = pipe.predict_single(image_path)
    except Exception as e:
        return {"error": f"파이프라인 실행 실패: {e}", "image_path": image_path}

    if result is None:
        return {"error": "얼굴을 검출하지 못했습니다.", "image_path": image_path}

    return {
        "image_path":      image_path,
        "gender_input":    gender,
        "age":             result.get("age") if gender_provided else None,
        "age_note":        None if gender_provided else "성별 미지정으로 피부 나이는 산출하지 않음. 사용자에게 성별을 확인 후 재호출 권장.",
        "valid_sagging":   result.get("valid_sagging"),
        "raw_scores":      _flatten(result),
    }
