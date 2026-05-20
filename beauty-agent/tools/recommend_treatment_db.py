"""state의 top_concerns를 받아 AuraDB FeatureRule에서 시술 추천을 조회하는 tool.

Phase 2 설계:
- 입력: 없음 (InjectedState로 state에서 top_concerns + skin_scores 읽음)
- 출력: state.recommendations 갱신 + LLM에 노출할 ToolMessage
- Hard guard: top_concerns가 없으면 skin_analyze를 먼저 호출하라는 에러 메시지를 반환
- 매칭: 우리 region 키 → DB feature_name 매핑, age 있으면 age_group 자동 매칭, vector search top-1
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from config import PROJECT_ROOT
from tools.skin_analyze import aggregate_regions

# auradb/ 를 sys.path에 추가해 Connect_DB의 함수를 재사용
AURADB_DIR = PROJECT_ROOT / "auradb"
if str(AURADB_DIR) not in sys.path:
    sys.path.insert(0, str(AURADB_DIR))


# 우리 진단 region 키 → AuraDB feature_name 매핑
REGION_TO_FEATURE = {
    "pigmentation":        "색소",
    "forehead_wrinkle":    "이마·미간주름",
    "eye_wrinkle":         "눈가·앞광대주름",
    "nasolabial_fold":     "팔자주름",
    "perioral_wrinkle":    "입가·턱주름",
    "volume_wrinkle":      "볼꺼짐",
    "homogenity_radiance": "스킨부스터 (new)",
    "homogenity_texture":  "스킨부스터 (new)",
    "cheek_sagging":       "리프팅(윤곽)",
    "chin_sagging":        "리프팅(윤곽)",
}


def _severity_label(score: float) -> str:
    if score <= 30:
        return "심각"
    if score <= 80:
        return "보통"
    return "경미"


def _parse_age_group(s: str) -> tuple[int, int]:
    """'29세 이하' → (0, 29), '30-39세' → (30, 39), '30세 이상' → (30, 999)."""
    s = (s or "").strip()
    m = re.match(r"(\d+)\s*세\s*이하", s)
    if m:
        return (0, int(m.group(1)))
    m = re.match(r"(\d+)\s*-\s*(\d+)\s*세", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"(\d+)\s*세\s*이상", s)
    if m:
        return (int(m.group(1)), 999)
    return (0, 999)


def _best_age_group(age: float, candidates: list[str]) -> Optional[str]:
    """age가 속하는 가장 좁은 범위의 age_group 선택. 후보 없으면 None."""
    a = int(round(age))
    fits = []
    for c in candidates:
        lo, hi = _parse_age_group(c)
        if lo <= a <= hi:
            fits.append((c, hi - lo))
    if not fits:
        return None
    fits.sort(key=lambda x: x[1])
    return fits[0][0]


def _make_tool_message(content, tool_call_id: str) -> ToolMessage:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=tool_call_id)

def _group_recommendations(recs: list[dict]) -> list[dict]:
    """feature_name 기준으로 묶고, 각 묶음의 대표 항목은 가장 낮은 score를 사용한다."""
    grouped: dict[str, dict] = {}

    for rec in recs:
        feature_name = rec.get("feature_name") or "미분류"
        group = grouped.setdefault(feature_name, {
            "feature_name": feature_name,
            "regions": [],
            "best": rec,
        })
        group["regions"].append(rec)
        if rec["score"] < group["best"]["score"]:
            group["best"] = rec

    return sorted(
        grouped.values(),
        key=lambda g: (g["best"]["score"], g["feature_name"]),
    )


def _format_region_list(regions: list[dict]) -> str:
    ordered = sorted(regions, key=lambda r: (r["score"], r["region_ko"]))
    return ", ".join(f"{r['region_ko']} {r['score']:.1f}점" for r in ordered)


def _format_group_block(group: dict, include_rank: bool = False, rank: int | None = None) -> list[str]:
    best = group["best"]
    feature_name = group["feature_name"]
    regions_text = _format_region_list(group["regions"])
    code = best.get("code", "") or ""
    treatment = (best.get("treatment") or "").strip()
    desc = (best.get("customer_desc") or "").replace("\n", " ").strip()

    header = f"- {feature_name} [{regions_text}]"
    if include_rank and rank is not None:
        header = f"{rank}. {feature_name} [{regions_text}]"

    lines = [header]
    if code.endswith("_0"):
        lines.append("  · 관리 불필요")
    else:
        lines.append(f"  · 권장 시술: {code}: {treatment}")
    if desc:
        lines.append(f"  · 안내: {desc}")
    return lines


def _format_recommendations(recs: list[dict], errors: list[dict]) -> str:
    """LLM에 노출할 자연어 요약. similarity/query_used/clinician_desc/matched_age_group 등 내부 값은 제외한다."""
    if not recs:
        body = "조회된 추천 결과가 없습니다."
    else:
        grouped = _group_recommendations(recs)
        top3 = grouped[:3]

        lines = ["AuraDB FeatureRule 조회 결과"]
        lines.append("\n[부위/카테고리별 권장 시술]")
        for group in grouped:
            lines.extend(_format_group_block(group))

        if top3:
            lines.append("\n[추천 시술 - 심각도가 높은 카테고리 3개]")
            for idx, group in enumerate(top3, start=1):
                lines.extend(_format_group_block(group, include_rank=True, rank=idx))

        body = "\n".join(lines)

    if errors:
        err_str = "; ".join(f"{e.get('region')}: {e.get('reason')}" for e in errors)
        body += f"\n\n[조회 오류] {err_str}"
    return body


def _build_query(region_ko: str, score: float, age: Optional[float]) -> str:
    """customer_desc 임베딩과 매칭되도록 자연어 쿼리 구성."""
    sev = _severity_label(score)
    age_part = f", 추정 연령 {int(round(age))}세" if age else ""
    return f"{region_ko} 점수 {score:.1f}점 ({sev}){age_part}"


@tool
def recommend_treatment_db(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """skin_analyze로 도출된 raw_scores를 받아 AuraDB FeatureRule에서 적절한 시술을 조회합니다.
    feature_name 기준으로 묶어서 안내하고, 그중 진단 점수가 가장 낮은 카테고리 3개를 별도 추천합니다.
    반드시 skin_analyze를 먼저 호출한 뒤 사용하세요. 추가 인자는 받지 않으며 state를 그대로 활용합니다.

    Returns:
        Command(update={...}) — state.db_recommendations 갱신 + ToolMessage 삽입.
    """
    skin_scores = state.get("skin_scores") or {}
    raw_scores = skin_scores.get("raw_scores")
    if not raw_scores:
        return Command(update={"messages": [_make_tool_message(
            "진단 데이터(raw_scores)가 없습니다. 먼저 skin_analyze를 호출하세요.",
            tool_call_id,
        )]})

    # 지연 import: DB 미연결 환경에서도 module load는 성공해야 함
    try:
        spec = importlib.util.spec_from_file_location(
            "_connect_db", str(AURADB_DIR / "Connect_DB.py")
        )
        Connect_DB = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Connect_DB)

        spec_rm = importlib.util.spec_from_file_location(
            "_rule_matcher", str(AURADB_DIR / "rule_matcher.py")
        )
        rule_matcher = importlib.util.module_from_spec(spec_rm)
        spec_rm.loader.exec_module(rule_matcher)
    except Exception as e:
        return Command(update={"messages": [_make_tool_message(
            {"error": f"auradb 모듈 로드 실패: {e}"},
            tool_call_id,
        )]})

    age = skin_scores.get("age")  # None이면 age_group 미사용

    # 전체 10개 region을 조회하되, 출력은 feature_name 기준으로 묶어서 정리
    all_regions = aggregate_regions(raw_scores)

    db_recommendations = []
    errors = []
    age_group_cache: dict = {}

    for concern in all_regions:
        region = concern["region"]
        region_ko = concern["region_ko"]
        score = concern["score"]
        feature_name = REGION_TO_FEATURE.get(region)
        if not feature_name:
            errors.append({"region": region, "reason": "feature_name 매핑 없음"})
            continue

        # age_group 결정 (feature_name별로 한 번만 조회 캐시)
        age_group = None
        if age is not None:
            if feature_name not in age_group_cache:
                try:
                    age_group_cache[feature_name] = Connect_DB.list_age_groups(feature_name)
                except Exception as e:
                    age_group_cache[feature_name] = []
                    errors.append({"region": region, "reason": f"age_group 조회 실패: {e}"})
            cands = age_group_cache.get(feature_name) or []
            age_group = _best_age_group(age, cands) if cands else None

        query_text = _build_query(region_ko, score, age)

        # rule-based 매칭 (1차)
        try:
            rows = Connect_DB.fetch_feature_rows(feature_name, age_group=age_group)
        except Exception as e:
            errors.append({"region": region, "reason": f"DB 조회 실패: {e}"})
            continue

        m = rule_matcher.match_feature_row(feature_name, age_group, raw_scores, rows)
        matched_by = "rule"

        # fallback: rule이 못 찾으면 vector 유사도
        if m is None:
            try:
                matches = Connect_DB.search_feature_by_name(
                    feature_name=feature_name,
                    query_text=query_text,
                    age_group=age_group,
                    top_k=1,
                )
                m = matches[0] if matches else None
                matched_by = "vector_fallback" if m else None
            except Exception as e:
                errors.append({"region": region, "reason": f"vector fallback 실패: {e}"})
                continue

        if not m:
            errors.append({"region": region, "reason": "조회 결과 없음"})
            continue

        db_recommendations.append({
            "region":            region,
            "region_ko":         region_ko,
            "score":             score,
            "feature_name":      feature_name,
            "matched_age_group": m.get("age_group"),
            "matched_by":        matched_by,
            "code":              m.get("code"),
            "treatment":         m.get("output_text"),
            "customer_desc":     m.get("customer_desc"),
            "clinician_desc":    m.get("clinician_desc"),
            "similarity":        m.get("score"),
            "query_used":        query_text,
        })

    # LLM이 보는 ToolMessage는 정제된 자연어. 내부값(similarity/query_used 등)은 state에만 둠.
    summary = _format_recommendations(db_recommendations, errors)

    return Command(update={
        "db_recommendations": db_recommendations,
        "messages":           [_make_tool_message(summary, tool_call_id)],
    })
