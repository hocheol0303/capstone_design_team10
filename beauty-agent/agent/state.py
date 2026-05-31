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
    iteration_count: Annotated[int, "loop 횟수 명시"]   # think 노드가 매번 +1
    max_iterations:  Annotated[int, "최대 loop 횟수"]   # 도달 시 finish로 강제 분기

    # ── 실행 컨텍스트 + 결과 ──────────────────────────────────
    current_goal: Annotated[str, "현재 달성하려는 목표"]
    intent:       Annotated[str | None, "classify_intent_node가 결정한 의도 (report | general)"]
    final_answer: Annotated[str, "최종 답변"]
    is_complete:  Annotated[bool, "목표 달성 여부 (True면 finish로 분기)"]

    # ── 도메인 컨텍스트 (skin agent 전용) ─────────────────────
    image_path:             Annotated[str | None, "이미지 경로"]
    gender:                 Annotated[str | None, "성별"]
    skin_scores:            Annotated[dict | None, "피부 점수"]
    top_concerns:           Annotated[list | None, "주요 관심사"]
    db_recommendations:     Annotated[list | None, "데이터베이스 추천"]
    pubmed_recommendations: Annotated[list | None, "PubMed 추천"]
    db_forced:              Annotated[bool, "진단 직후 시술 추천(DB) 조회를 1회 강제했는지"]
    pubmed_forced:          Annotated[bool, "추천 직후 PubMed 근거 검색을 1회 강제했는지"]
    compressed_summary:     Annotated[str | None, "compress 노드가 만든 추천/근거 압축 텍스트"]
