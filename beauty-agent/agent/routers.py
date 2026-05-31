"""Beauty agent LangGraph 라우팅 함수 모음.

각 함수는 state를 읽고 다음 노드 이름(문자열)을 반환한다.
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from agent.helpers import DEFAULT_MAX_ITERATIONS, latest_human_message_text
from agent.state import BeautyAgentState

# 추천 의도로 볼 키워드 (현재 턴의 사용자 메시지 기준 휴리스틱)
_RECOMMEND_HINT = re.compile(
    r"(추천|시술|treatment|recommend|케어|관리|받|뭐가\s*좋|필요)", re.IGNORECASE
)


def _wants_recommendation(state: BeautyAgentState) -> bool:
    # current_goal은 세션 첫 메시지에 1회만 고정되므로, 현재 턴 입력(최신 HumanMessage)으로 판단한다.
    latest = latest_human_message_text(state.get("messages") or [])
    return bool(_RECOMMEND_HINT.search(latest))


def route_after_think(state: BeautyAgentState) -> str:
    """think이 tool_call을 발행했으면 act, 아니면 곧장 finish.

    ToolNode가 tool_call 없는 메시지에서 에러를 낼 수 있어 안전 분기.
    (추천 강제는 observe 이후 should_continue에서 다음 think 전에 끼어들어 처리한다.
     여기서 강제하면 think이 이미 스트리밍한 텍스트 뒤에 답이 중복되므로 하지 않는다.)
    """
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "act"
    return "finish"


def should_continue(state: BeautyAgentState) -> str:
    """observe 이후: 루프 계속 / 근거 검색 강제 / 종료 결정.

    - iteration_count가 max_iterations에 도달했으면 finish (loop guard).
    - 시술 추천(db_recommendations)은 나왔는데 PubMed 근거가 아직 없고
      아직 강제 검색을 안 했다면 inject_pubmed로 분기해 근거를 확보한다.
    - 그 외에는 think로 돌아가 LLM이 추가 행동 또는 최종 답변을 결정.
    """
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if iter_count >= max_iter:
        return "finish"
    # 진단은 됐는데 추천(db)이 아직 없고, 추천 의도이며, 아직 강제 안 했다면 → 추천 조회 강제.
    if (
        (state.get("skin_scores") or {}).get("raw_scores")
        and not state.get("db_recommendations")
        and not state.get("db_forced")
        and _wants_recommendation(state)
    ):
        return "inject_recommend"
    # 추천(db)은 나왔는데 PubMed 근거가 아직 없고 아직 강제 안 했다면 → 근거 검색 강제.
    if (
        state.get("db_recommendations")
        and not state.get("pubmed_recommendations")
        and not state.get("pubmed_forced")
    ):
        return "inject_pubmed"
    return "think"


def route_after_classify(state: BeautyAgentState) -> str:
    """classify_intent_node 이후 1차 분기.

    - intent == 'report' → data_gate (데이터 검증 단계로)
    - 그 외 → think (일반 ReAct)
    """
    if state.get("intent") == "report":
        return "data_gate"
    return "think"


def route_after_data_gate(state: BeautyAgentState) -> str:
    """data_gate 이후 2차 분기. report 의도가 확정된 상태에서 데이터 충분 여부 검증.

    - 진단 + (시술 추천 or 논문 근거) → compress (→ final_report)
    - 부족 → insufficient_response
    """
    has_diag = bool((state.get("skin_scores") or {}).get("raw_scores"))
    has_db = bool(state.get("db_recommendations"))
    has_pub = bool(state.get("pubmed_recommendations"))
    if has_diag and (has_db or has_pub):
        return "compress"
    return "insufficient_response"
