"""시나리오 2: 확인형 - '저는 색소 치료를 받아야 하나요?'"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_agent
from tests._assertions import tool_call_sequence, tool_observations


def test_scenario_2():
    os.environ["USE_MOCK_SCENARIO"] = "2"

    image_path = "tests/mock_images/scenario_2.jpg"
    text = "저는 색소 치료를 받아야 하나요?"

    result = run_agent(image_path, text)
    final = result["final_answer"]
    sequence = tool_call_sequence(result["messages"])
    skin_obs = tool_observations(result["messages"], "skin_analyze")

    print("=== Final Answer ===")
    print(final)
    print("=== Tool Call Sequence ===")
    print(sequence)

    # 1. skin_analyze 결과의 pigmentation이 mild인지 확인
    assert skin_obs, "skin_analyze ToolMessage가 없습니다."
    skin_payload = skin_obs[0].content
    if isinstance(skin_payload, str):
        try:
            skin_payload = json.loads(skin_payload.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    payload_text = str(skin_payload)
    assert "pigmentation" in payload_text and "mild" in payload_text, \
        f"pigmentation이 mild로 판단되지 않았습니다: {payload_text}"

    # 2. 'mild pigmentation' 관련 search_pubmed 호출
    assert "search_pubmed" in sequence, "search_pubmed가 호출되지 않았습니다."

    # 3. Final Answer에 색소 치료 불필요 또는 유사한 내용
    keywords = ["불필요", "필요하지", "권장하지", "근거가 부족", "mild", "경미"]
    assert any(k in final for k in keywords), \
        "Final Answer에 색소 치료가 불필요하다는 내용이 보이지 않습니다."

    # 4. 이마주름/팔자주름 관련 추가 소견
    additional = ["이마주름", "팔자주름", "이마 주름", "팔자 주름", "forehead", "nasolabial"]
    assert any(k in final for k in additional), \
        "Final Answer에 이마주름/팔자주름 관련 추가 소견이 없습니다."

    # 5. '전문의' 문구 포함
    assert "전문의" in final, "Final Answer에 '전문의' 문구가 없습니다."

    print("\n[PASS] 시나리오 2 모든 검증 통과")


if __name__ == "__main__":
    test_scenario_2()
