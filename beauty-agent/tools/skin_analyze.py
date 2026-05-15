from langchain_core.tools import tool

from config import USE_MOCK_SCENARIO, USE_MOCK_VISION

_MOCK_SCENARIO_1 = {
    "pigment_left":       78,
    "pigment_right":      84,
    "wrinkle_forehead":   18,
    "wrinkle_nasolabial": 22,
    "wrinkle_right_eye":  45,
    "wrinkle_left_eye":   48,
    "wrinkle_perioral":   35,
    "sagging_jawline":    55,
    "volume_loss":        40,
}

_MOCK_SCENARIO_2 = _MOCK_SCENARIO_1

_MOCK_SCENARIO_3 = {
    "pigment_left":       22,
    "pigment_right":      25,
    "wrinkle_forehead":   72,
    "wrinkle_nasolabial": 68,
    "wrinkle_right_eye":  55,
    "wrinkle_left_eye":   58,
    "wrinkle_perioral":   45,
    "sagging_jawline":    80,
    "volume_loss":        62,
}

_MOCK_REGISTRY = {
    1: _MOCK_SCENARIO_1,
    2: _MOCK_SCENARIO_2,
    3: _MOCK_SCENARIO_3,
}

_REGION_LABEL_KO = {
    "pigmentation":      "색소침착",
    "forehead_wrinkle":  "이마주름",
    "nasolabial_fold":   "팔자주름",
    "eye_wrinkle":       "눈가주름",
    "perioral_wrinkle":  "입가주름",
    "jawline_sagging":   "턱선 처짐",
    "volume_loss":       "볼륨 손실",
}


def score_to_severity(score: float) -> str:
    if score <= 30:
        return "severe"
    elif score <= 80:
        return "moderate"
    else:
        return "mild"


def _aggregate(raw: dict) -> dict:
    pigmentation = (raw["pigment_left"] + raw["pigment_right"]) / 2
    eye_wrinkle = (raw["wrinkle_right_eye"] + raw["wrinkle_left_eye"]) / 2
    return {
        "pigmentation":     pigmentation,
        "forehead_wrinkle": raw["wrinkle_forehead"],
        "nasolabial_fold":  raw["wrinkle_nasolabial"],
        "eye_wrinkle":      eye_wrinkle,
        "perioral_wrinkle": raw["wrinkle_perioral"],
        "jawline_sagging":  raw["sagging_jawline"],
        "volume_loss":      raw["volume_loss"],
    }


def _build_output(raw_scores: dict) -> dict:
    aggregated = _aggregate(raw_scores)
    severity = {region: score_to_severity(s) for region, s in aggregated.items()}

    severity_rank = {"severe": 0, "moderate": 1, "mild": 2}
    primary_concerns = [
        f"{region} ({sev})"
        for region, sev in sorted(severity.items(), key=lambda kv: severity_rank[kv[1]])
        if sev == "severe"
    ]

    severity_summary = ", ".join(
        f"{_REGION_LABEL_KO[region]}({sev})" for region, sev in severity.items()
    )

    return {
        "severity": severity,
        "primary_concerns": primary_concerns,
        "severity_summary": severity_summary,
    }


@tool
def skin_analyze(image: str) -> dict:
    """얼굴 이미지를 분석하여 피부 상태를 부위별 중증도로 산출합니다 (severe/moderate/mild). 이미지가 입력되면 반드시 첫 번째로 호출하세요.

    Args:
        image: Base64 encoded 얼굴 이미지 또는 이미지 경로.

    Returns:
        부위별 severity, primary_concerns, severity_summary를 담은 dict.
    """
    if USE_MOCK_VISION:
        raw_scores = _MOCK_REGISTRY.get(USE_MOCK_SCENARIO, _MOCK_SCENARIO_1)
        return _build_output(raw_scores)

    raise NotImplementedError("실제 비전 모델 연동은 추후 Phase에서 구현합니다.")
