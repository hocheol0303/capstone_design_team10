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

DEFAULT_MAX_ITERATIONS = 6


def _load_llm():
    return init_chat_model(model=AI_MODEL, temperature=0)


def _latest_human_message_text(messages: list[BaseMessage]) -> str:
    """가장 최근 HumanMessage의 텍스트. 없으면 빈 문자열."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _parse_image_path(text: str) -> str | None:
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


def _parse_gender(text: str) -> str | None:
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


def _classify_intent(user_text: str) -> str:
    """LLM 기반 의도 분류. 'report' 또는 'general'.

    호출 실패 또는 입력이 비어 있으면 안전하게 'general' 반환.
    """
    if not user_text or not user_text.strip():
        return "general"
    try:
        llm = _load_llm()
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=f"사용자 메시지: {user_text}\n분류:"),
        ])
        label = _extract_text(getattr(response, "content", "")).strip().lower()
        return "report" if "report" in label else "general"
    except Exception:  # noqa: BLE001
        return "general"


def _extract_text(content: Any) -> str:
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


def _build_system_context(state: BeautyAgentState) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "현재 state 요약:\n"
        f"- current_goal: {state.get('current_goal') or '(아직 설정되지 않음)'}\n"
        f"- image_path: {state.get('image_path') or 'None'}\n"
        f"- gender: {state.get('gender') or 'None'}\n"
        f"- skin_scores: {'존재' if state.get('skin_scores') else '없음'}\n"
        f"- db_recommendations: {'존재' if state.get('db_recommendations') else '없음'}\n"
        f"- pubmed_recommendations: {'존재' if state.get('pubmed_recommendations') else '없음'}\n"
        f"- iteration: {state.get('iteration_count', 0)}/{state.get('max_iterations', DEFAULT_MAX_ITERATIONS)}\n"
    )


def _extract_state_context(state: BeautyAgentState) -> dict[str, str | None]:
    """가장 최근 HumanMessage(text)에서 image_path/gender를 파싱해 state에 정착시킨다."""
    text = _latest_human_message_text(state.get("messages") or [])
    image_path = _parse_image_path(text)
    gender = _parse_gender(text)
    updates: dict[str, str | None] = {}
    if image_path:
        updates["image_path"] = image_path
    if gender:
        updates["gender"] = gender
    return updates


def _format_tool_args(args: dict | None) -> str:
    if not args:
        return ""
    pieces = []
    for key, value in args.items():
        if isinstance(value, str):
            pieces.append(f"{key}={value!r}")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def _reinput_request_message() -> str:
    return "입력에 처리할 수 없는 문자가 포함되어 있습니다. 다시 입력해 주세요."
