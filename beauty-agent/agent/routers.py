"""Beauty agent LangGraph 라우팅 함수 모음.

각 함수는 state를 읽고 다음 노드 이름(문자열)을 반환한다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.helpers import DEFAULT_MAX_ITERATIONS
from agent.state import BeautyAgentState

# think 노드에서 tool call을 발행했는지 보고 act/finish으로 라우팅하는 함수
def route_after_think(state: BeautyAgentState) -> str:
    """think이 tool_call을 발행했으면 act, 아니면 곧장 finish.

    ToolNode가 tool_call 없는 메시지에서 에러를 낼 수 있어 안전 분기.
    """
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        # tool_calls가 있으면 act로 라우팅해서 도구 호출 후 내용 Observe
        return "act"
    return "finish"


def should_continue(state: BeautyAgentState) -> str:
    """observe 이후: 루프 계속 / 종료 결정.

    - iteration_count가 max_iterations에 도달했으면 finish (loop guard).
    - 그 외에는 think로 돌아가 LLM이 추가 행동 또는 최종 답변을 결정.
    """
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if iter_count >= max_iter:
        return "finish"
    return "think"

# 첫 입력 이후 classify_intent 노드에서 분류한 intent state를 보고 다음 노드로 라우팅하는 함수 (=조건문)
def route_after_classify(state: BeautyAgentState) -> str:
    """classify_intent_node 이후 1차 분기.

    - intent == 'report' → data_gate (데이터 검증 단계로)
    - 그 외 → think (일반 ReAct)
    """
    if state.get("intent") == "report":
        return "data_gate"
    return "think"

# classify_intent_node 이후 report 의도가 확정된 상태에서 레포트를 작성하기에 데이터가 충분한지 여부를 검증하는 함수
# 진단을 받았는지 여부와 [시술 추천 or 논문 검색]이 있으면 데이터가 충분하다고 보고 compress로, 아니면 insufficient_response로 라우팅하여 데이터 더 쌓으라고 요청한다.한다.
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
