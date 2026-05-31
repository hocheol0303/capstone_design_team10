"""Beauty agent 공용 헬퍼.

graph.py에서 분리된 순수 함수 모음:
- LLM 로더, 메시지 파서, 의도 분류, system context 빌더, 텍스트 추출 등.
- 상태에 부수효과 없음 (state 읽기만).
"""
from __future__ import annotations

import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import BeautyAgentState
from config import AI_MODEL
from tools.skin_analyze import aggregate_regions

DEFAULT_MAX_ITERATIONS = 6


def load_llm(
    temperature: float = 0.0,
    frequency_penalty: float = 0.0,
    max_tokens: int | None = None,
):
    """LLM 로더. 기본은 결정적(temperature=0).

    대화형 노드(think/final_report)는 친근한 말투를 위해 약간 높은 온도를 넘기고,
    긴 리포트 생성 시 반복 루프를 막기 위해 frequency_penalty/max_tokens를 함께 준다.
    JSON을 뱉는 내부 단계(분류/검색어/리랭킹)는 기본값을 그대로 쓴다.
    """
    kwargs: dict = {"model": AI_MODEL, "temperature": temperature}
    if frequency_penalty:
        kwargs["frequency_penalty"] = frequency_penalty
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)


def latest_human_message_text(messages: list[BaseMessage]) -> str:
    """가장 최근 HumanMessage의 텍스트. 없으면 빈 문자열."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def parse_image_path(text: str) -> str | None:
    patterns = [
        r"(?:image_path|path|경로)\s*[:=]\s*([^\s,]+)",
        r"(?:image|img)\s*[:=]\s*([^\s,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("'\"")
    inline_path = re.search(r"([^\s]+\.(?:jpg|jpeg|png|webp|bmp|gif))", text, re.IGNORECASE)
    if inline_path:
        return inline_path.group(1).strip().strip("'\"")
    return None


def parse_gender(text: str) -> str | None:
    normalized = text.lower()
    if re.search(r"(?:\b여자\b|\bfemale\b|\bwoman\b)", normalized):
        return "female"
    if re.search(r"(?:\b남자\b|\bmale\b|\bman\b)", normalized):
        return "male"
    return None


_INTENT_SYSTEM = (
    "당신은 한국어 사용자 메시지를 두 카테고리로 분류하는 의도 분류기입니다. "
    "정확히 'report' 또는 'general' 한 단어로만 답변하세요.\n"
    "- report: 누적된 진단/추천 데이터를 종합한 최종 보고서·요약·정리·총평을 요청하는 의도. "
    "예) '최종 레포트 작성해줘', '지금까지 내용 정리해줘', '총정리 부탁', '요약 보고서 만들어줘'.\n"
    "- general: 그 외 모든 의도. 예) 진단 요청, 시술 추천 요청, 일반 대화, 후속 질문 등."
)


def classify_intent(user_text: str) -> str:
    """LLM 기반 의도 분류. 'report' 또는 'general'.

    호출 실패 또는 입력이 비어 있으면 안전하게 'general' 반환.
    """
    if not user_text or not user_text.strip():
        return "general"
    try:
        llm = load_llm()
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=f"사용자 메시지: {user_text}\n분류:"),
        ])
        label = extract_text(getattr(response, "content", "")).strip().lower()
        return "report" if "report" in label else "general"
    except Exception:  # noqa: BLE001
        return "general"


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def format_current_scores(state: BeautyAgentState) -> str:
    """state에 저장된 raw_scores를 부위별 집계 점수로 정리해 system context에 노출.

    후속 질문(예: '색소는 어때?')에서 LLM이 기억 대신 실제 값을 근거로
    '낮을수록 심각' 규칙에 맞게 답하도록 사실을 명시적으로 제공한다.
    """
    skin_scores = state.get("skin_scores") or {}
    raw_scores = skin_scores.get("raw_scores")
    if not raw_scores:
        return "없음"

    aggregated = aggregate_regions(raw_scores)
    if not aggregated:
        return "존재(점수 파싱 불가)"

    score_lines = ", ".join(f"{r['region_ko']} {r['score']:.1f}점" for r in aggregated)
    worst = ", ".join(f"{r['region_ko']} {r['score']:.1f}점" for r in aggregated[:3])
    age = skin_scores.get("age")
    age_part = f" / 추정 연령 {age:.0f}세" if isinstance(age, (int, float)) else ""
    return (
        f"존재{age_part}\n"
        f"    · 부위별 점수(낮을수록 심각, 높을수록 양호): {score_lines}\n"
        f"    · 가장 심각한(점수 낮은) 부위 top3: {worst}"
    )


def severe_untreated_directive(state: BeautyAgentState) -> str:
    """추천에서 빠진 '최악 부위' 브리지 문장을 결정적으로 계산해 반환.

    db_recommendations 중 코드가 _0(시술 불필요)인데, 정작 점수가 '추천된 부위
    중 가장 낮은 점수'보다도 더 낮은(=더 심각한) 부위를 찾는다. 이런 부위는
    Step 1에서 최악으로 강조됐다가 Step 2에서 소리 없이 사라져 사용자를 혼란스럽게
    하므로, 모델이 거의 그대로 베껴 쓸 완성된 한 줄을 미리 만들어 준다.

    조건에 해당하는 부위가 없으면 빈 문자열을 반환한다.
    """
    db_recs = state.get("db_recommendations") or []
    if not db_recs:
        return ""

    def _is_untreated(rec: dict) -> bool:
        return (rec.get("code") or "").endswith("_0")

    needs = [r for r in db_recs if not _is_untreated(r) and r.get("score") is not None]
    untreated = [r for r in db_recs if _is_untreated(r) and r.get("score") is not None]
    if not needs or not untreated:
        return ""

    worst_needed_score = min(r["score"] for r in needs)
    flagged = sorted(
        (r for r in untreated if r["score"] < worst_needed_score),
        key=lambda r: r["score"],
    )[:3]
    if not flagged:
        return ""

    listed = ", ".join(f"{r['region_ko']} {r['score']:.1f}점" for r in flagged)
    names = "·".join(r["region_ko"] for r in flagged)
    sentence = (
        f"가장 점수가 낮은 {names}는 현재 DB상 바로 권장되는 시술은 없으나, "
        "보습·자외선 차단 등 생활관리와 함께 대면 상담에서 별도로 살펴보시길 권합니다."
    )
    return (
        f"[필수 포함 — 최악 부위 브리지] 점수가 가장 낮은 {listed}이(가) DB상 '시술 불필요(_0)'로 "
        "분류되어 Step 2 추천 카드에서 빠집니다. 사용자가 자신의 최악 부위가 이유 없이 사라졌다고 "
        "느끼지 않도록, Step 2 마지막에 아래 문장을 거의 그대로 한 줄 넣으세요(생략 금지):\n"
        f"      \"{sentence}\""
    )


def build_system_context(state: BeautyAgentState) -> str:
    directive = severe_untreated_directive(state)
    directive_block = f"\n{directive}\n" if directive else ""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "현재 state 요약:\n"
        f"- current_goal: {state.get('current_goal') or '(아직 설정되지 않음)'}\n"
        f"- image_path: {state.get('image_path') or 'None'}\n"
        f"- gender: {state.get('gender') or 'None'}\n"
        f"- skin_scores: {format_current_scores(state)}\n"
        f"- db_recommendations: {'존재' if state.get('db_recommendations') else '없음'}\n"
        f"- pubmed_recommendations: {'존재' if state.get('pubmed_recommendations') else '없음'}\n"
        f"- iteration: {state.get('iteration_count', 0)}/{state.get('max_iterations', DEFAULT_MAX_ITERATIONS)}\n"
        f"{directive_block}"
    )


def extract_state_context(state: BeautyAgentState) -> dict[str, str | None]:
    """가장 최근 HumanMessage(text)에서 image_path/gender를 파싱해 state에 정착시킨다."""
    text = latest_human_message_text(state.get("messages") or [])
    image_path = parse_image_path(text)
    gender = parse_gender(text)
    updates: dict[str, str | None] = {}
    if image_path:
        updates["image_path"] = image_path
    if gender:
        updates["gender"] = gender
    return updates


def format_tool_args(args: dict | None) -> str:
    if not args:
        return ""
    pieces = []
    for key, value in args.items():
        if isinstance(value, str):
            pieces.append(f"{key}={value!r}")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def reinput_request_message() -> str:
    return "입력에 처리할 수 없는 문자가 포함되어 있습니다. 다시 입력해 주세요."
