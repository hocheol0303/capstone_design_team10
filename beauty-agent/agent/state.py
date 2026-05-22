"""Beauty agent용 LangGraph state schema (책 정석 ReAct + 도메인 확장).

기반: 위키독스 LangGraph part 2-1-2의 표준 ReAct AgentState.
- 추론 흔적 3종: thoughts / actions / observations (append reducer)
- 루프 가드: iteration_count + max_iterations
- 실행 컨텍스트: current_goal
- 결과: final_answer + is_complete

추가(도메인): image_path, gender, skin_scores, top_concerns,
              db_recommendations, pubmed_recommendations.

messages는 MessagesState가 제공(add_messages reducer).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from langgraph.graph import MessagesState


class BeautyAgentState(MessagesState):
    """LangGraph state for the Beauty Agent (책 정석 ReAct 패턴)."""

    # ── 책 정석 ReAct 추론 흔적 ───────────────────────────────
    thoughts:     Annotated[List[str], operator.add]
    actions:      Annotated[List[Dict[str, Any]], operator.add]
    observations: Annotated[List[str], operator.add]

    # ── 루프 가드 ────────────────────────────────────────────
    iteration_count: int   # think 노드가 매번 +1
    max_iterations:  int   # 도달 시 finish로 강제 분기

    # ── 실행 컨텍스트 + 결과 ──────────────────────────────────
    current_goal: str
    final_answer: str
    is_complete:  bool

    # ── 도메인 컨텍스트 (skin agent 전용) ─────────────────────
    image_path:             str | None
    gender:                 str | None
    skin_scores:            dict | None
    top_concerns:           list | None
    db_recommendations:     list | None
    pubmed_recommendations: list | None
