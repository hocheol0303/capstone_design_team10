"""Beauty agent용 LangGraph state schema (단일 소스)."""
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class BeautyAgentState(TypedDict, total=False):
    """LangGraph state for the Beauty Agent.

    - messages: 대화 메시지 히스토리 (append)
    - image_path: 진단 대상 이미지 경로
    - gender: 사용자 성별 ('male'/'female'/None)
    - skin_scores: skin_analyze 결과 dict (image_path, gender_input, age, age_note,
                   valid_sagging, raw_scores)
    - top_concerns: 점수가 가장 낮은(=가장 심각한) 부위 3개 (skin_analyze 산출)
    - db_recommendations: recommend_treatment_db 결과 리스트 (region별 dict)
    - reasoning_steps: ReAct 단계 로그 (append). 디버깅/UI 노출용.
    """

    messages: Annotated[list[BaseMessage], add]
    image_path: str | None
    gender: str | None
    skin_scores: dict | None
    top_concerns: list | None
    db_recommendations: list | None
    reasoning_steps: Annotated[list[str], add]
