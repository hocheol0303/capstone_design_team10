"""Beauty agent용 커스텀 state schema.

`langchain.agents.create_agent`는 기본적으로 messages만 가진 AgentState를 쓴다.
피부 진단 결과를 일급 state 필드로 들고 다니기 위해 확장 schema를 정의한다.
"""
from typing import Optional

from langchain.agents import AgentState


class BeautyAgentState(AgentState):
    """Beauty agent state.

    - skin_scores: skin_analyze tool이 반환한 전체 진단 결과(dict). 한 번 채워지면 유지.
    - top_concerns: 가장 점수가 낮은(=상태가 심각한) 부위 상위 3개 (LLM 진단 요약용).
        예) [{"region": "nasolabial_fold", "region_ko": "팔자주름", "score": 44.00}, ...]
    - db_recommendations: recommend_treatment가 AuraDB에서 조회한 부위별 시술 결과.
        각 항목은 region, feature_name, code, treatment(output_text), customer_desc, similarity 등을 포함.
    """

    skin_scores: Optional[dict]
    top_concerns: Optional[list]
    db_recommendations: Optional[list]
