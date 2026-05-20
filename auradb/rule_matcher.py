"""Rule-based FeatureRule 매칭.

DB row의 condition_1/2/3 텍스트를 raw 점수로 결정론적으로 평가한다.
4가지 condition 패턴:

(a) 단순 범위         예: "49점 이하", "50-64점", "80점 이상"
(b) 양쪽 비교         예: "양쪽 볼 모두 90점 이상", "한쪽 볼에서 59점 이하",
                       "양쪽 볼 모두 60점 이상이면서\n한쪽 볼에서 89점 이하"
(c) prefixed range    예: "턱처짐 35-64점", "볼꺼짐 54점 이하"
(d) 가중합 + 범위     예: "볼처짐 + 팔자주름 + 0.5*입가·턱주름 115-250점"

호출자(recommend_treatment_db)는 feature_name과 raw_scores(13개 키 dict)와
DB의 row 목록을 넘기면, match_feature_row가 어떤 row가 맞는지 결정한다.
"""
from __future__ import annotations

import re

# 한국어 변수명 → raw_scores key (= tools.skin_analyze._flatten 출력 키)
KO_VAR_TO_KEY = {
    "턱처짐":      "chin_sagging_total",
    "볼처짐":      "cheek_sagging_total",
    "팔자주름":    "wrinkle_nasolabial",
    "입가·턱주름": "wrinkle_perioral",
    "볼꺼짐":      None,  # 특수: wrinkle_left_vol + wrinkle_right_vol 평균
}


def _get_var(raw_scores: dict, ko_var: str) -> float | None:
    if ko_var == "볼꺼짐":
        lv = raw_scores.get("wrinkle_left_vol")
        rv = raw_scores.get("wrinkle_right_vol")
        vals = [v for v in (lv, rv) if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    key = KO_VAR_TO_KEY.get(ko_var)
    if not key:
        return None
    return raw_scores.get(key)


# ───────────────────────── (a) 단순 범위 ─────────────────────────

_RE_RANGE_AT_MOST  = re.compile(r"(\d+)\s*점\s*이하")
_RE_RANGE_AT_LEAST = re.compile(r"(\d+)\s*점\s*이상")
_RE_RANGE_BETWEEN  = re.compile(r"(\d+)\s*[-~]\s*(\d+)\s*점")


def parse_simple_range(text: str) -> tuple[float, float] | None:
    """'49점 이하' → (0,49), '50-64점' → (50,64), '80점 이상' → (80,100).
    매칭 실패 시 None.
    """
    if not text:
        return None
    s = text.strip()
    m = _RE_RANGE_BETWEEN.search(s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:  # "75-10점" 같은 오타는 None 반환 (호출자가 skip하게)
            return None
        return (float(lo), float(hi))
    m = _RE_RANGE_AT_MOST.search(s)
    if m:
        return (0.0, float(m.group(1)))
    m = _RE_RANGE_AT_LEAST.search(s)
    if m:
        return (float(m.group(1)), 100.0)
    return None


def _in_range(value: float, rng: tuple[float, float]) -> bool:
    lo, hi = rng
    return lo <= value <= hi


# ───────────────────────── (b) 양쪽 비교 ─────────────────────────

def eval_bilateral(text: str, left: float, right: float) -> bool:
    """양쪽/한쪽 표현 평가. 텍스트가 '이면서' 또는 줄바꿈으로 두 절 결합되어 있으면
    각 절을 평가한 뒤 AND.
    """
    if not text or left is None or right is None:
        return False
    s = text.replace("\n", " ").strip()
    # "...이면서..." 또는 ", " 로 합쳐진 다단계 조건은 AND
    clauses = re.split(r"이면서|,\s+", s)
    for clause in clauses:
        if not _eval_bilateral_clause(clause.strip(), left, right):
            return False
    return True


def _eval_bilateral_clause(clause: str, left: float, right: float) -> bool:
    rng = parse_simple_range(clause)
    if rng is None:
        return False
    if "양쪽" in clause:
        # both must fall in range
        return _in_range(left, rng) and _in_range(right, rng)
    if "한쪽" in clause:
        # at least one must fall in range
        return _in_range(left, rng) or _in_range(right, rng)
    # scope 미명시: 둘 다로 가정 (방어적으로 false)
    return False


# ───────────────────────── (c) prefixed range ─────────────────────────

def eval_prefixed_range(text: str, raw_scores: dict) -> bool:
    """'턱처짐 35-64점' → 턱처짐 점수가 [35,64] 안인지.

    text가 '없음' / 빈문자열 / parse 실패면 False.
    호출자가 None vs False를 구분해야 한다면 별도 함수로 분기.
    """
    if not text or text.strip() == "없음":
        return False
    s = text.strip()
    # 어떤 변수인지 식별
    for ko_var in KO_VAR_TO_KEY:
        if s.startswith(ko_var):
            tail = s[len(ko_var):].strip()
            rng = parse_simple_range(tail)
            if rng is None:
                return False
            val = _get_var(raw_scores, ko_var)
            if val is None:
                return False
            return _in_range(val, rng)
    return False


def is_none_marker(text: str) -> bool:
    """condition 텍스트가 '없음' 또는 비어있어 단계 통과로 봐야 하는지."""
    if text is None:
        return True
    return text.strip() in ("", "없음")


# ───────────────────────── (d) 가중합 + 범위 ─────────────────────────

# 변수 토큰 매칭: 한국어 변수명 + 가운데점 허용
_VAR_TOKEN = "|".join(re.escape(v) for v in KO_VAR_TO_KEY)
# '0.5*입가·턱주름' 또는 '입가·턱주름' 또는 '0.5 입가·턱주름'
_RE_TERM = re.compile(rf"(?:(\d+(?:\.\d+)?)\s*\*\s*)?({_VAR_TOKEN})")


def eval_weighted_range(text: str, raw_scores: dict) -> bool:
    """'볼처짐 + 팔자주름 + 0.5*입가·턱주름 115-250점'을 평가.
    가중치 없는 변수는 1.0. 끝부분에서 range 추출.
    """
    if not text or text.strip() == "없음":
        return False
    s = text.replace("\n", " ").strip()

    # 끝부분의 range를 떼어냄 (parse_simple_range가 검색하므로 그대로 사용)
    rng = parse_simple_range(s)
    if rng is None:
        return False

    # 변수 토큰 추출
    terms = _RE_TERM.findall(s)
    if not terms:
        return False

    total = 0.0
    for coef_str, ko_var in terms:
        coef = float(coef_str) if coef_str else 1.0
        val = _get_var(raw_scores, ko_var)
        if val is None:
            return False
        total += coef * val

    return _in_range(total, rng)


# ───────────────────────── feature dispatcher ─────────────────────────

# 단순 범위 feature → raw_scores key
_SIMPLE_RANGE_FEATURE_TO_KEY = {
    "이마·미간주름":  "wrinkle_forehead",
    "팔자주름":       "wrinkle_nasolabial",
    "입가·턱주름":    "wrinkle_perioral",
}

# 양쪽 비교 feature → (left key, right key)
_BILATERAL_FEATURE_TO_KEYS = {
    "색소":             ("pigment_left", "pigment_right"),
    "눈가·앞광대주름":  ("wrinkle_left_eye", "wrinkle_right_eye"),
    "볼꺼짐":           ("wrinkle_left_vol", "wrinkle_right_vol"),
}


def _eval_lifting_condition(cond: str, raw_scores: dict) -> bool:
    """리프팅(윤곽/탄력)의 condition 셀은 두 가지 중 하나:
    - prefixed range ("턱처짐 35-64점", "볼꺼짐 54점 이하")
    - weighted_range ("볼처짐 + 0.5*입가·턱주름 105-150점")
    """
    if is_none_marker(cond):
        return True
    # '+' 또는 '*' 가 있으면 weighted_range
    if "+" in cond or "*" in cond:
        return eval_weighted_range(cond, raw_scores)
    return eval_prefixed_range(cond, raw_scores)


def match_feature_row(
    feature_name: str,
    age_group: str | None,
    raw_scores: dict,
    rows: list[dict],
) -> dict | None:
    """후보 rows 중 feature_name 패턴에 맞춰 condition을 평가해 일치하는 row 반환."""
    if not rows:
        return None

    # age_group 필터 (None이면 무필터)
    candidates = [r for r in rows if not age_group or r.get("age_group") == age_group]
    if not candidates:
        candidates = rows  # age_group 매칭 실패해도 fallback

    if feature_name in _SIMPLE_RANGE_FEATURE_TO_KEY:
        key = _SIMPLE_RANGE_FEATURE_TO_KEY[feature_name]
        val = raw_scores.get(key)
        if val is None:
            return None
        for r in candidates:
            rng = parse_simple_range(r.get("condition_1") or "")
            if rng and _in_range(val, rng):
                return r
        return None

    if feature_name in _BILATERAL_FEATURE_TO_KEYS:
        lk, rk = _BILATERAL_FEATURE_TO_KEYS[feature_name]
        l = raw_scores.get(lk)
        r_val = raw_scores.get(rk)
        if l is None or r_val is None:
            return None
        for r in candidates:
            if eval_bilateral(r.get("condition_1") or "", l, r_val):
                return r
        return None

    if feature_name == "스킨부스터 (new)":
        radiance = raw_scores.get("homogenity_radiance")
        texture = raw_scores.get("homogenity_texture")
        if radiance is None or texture is None:
            return None
        for r in candidates:
            r1 = parse_simple_range(r.get("condition_1") or "")
            r2 = parse_simple_range(r.get("condition_2") or "")
            if r1 and r2 and _in_range(radiance, r1) and _in_range(texture, r2):
                return r
        return None

    if feature_name in {"리프팅(윤곽)", "리프팅(탄력)"}:
        for r in candidates:
            c1_ok = _eval_lifting_condition(r.get("condition_1") or "", raw_scores)
            c2_ok = _eval_lifting_condition(r.get("condition_2") or "", raw_scores)
            c3_ok = _eval_lifting_condition(r.get("condition_3") or "", raw_scores)
            if c1_ok and c2_ok and c3_ok:
                return r
        return None

    # 알 수 없는 feature
    return None
