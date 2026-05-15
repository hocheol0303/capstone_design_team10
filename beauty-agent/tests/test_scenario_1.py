"""시나리오 1: 진단형 - '어떤 시술을 받아야 할까요?'"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_agent
from tests._assertions import tool_call_sequence


def test_scenario_1():
    os.environ["USE_MOCK_SCENARIO"] = "1"

    image_path = "tests/mock_images/scenario_1.jpg"
    text = "어떤 시술을 받아야 할까요?"

    result = run_agent(image_path, text)
    final = result["final_answer"]
    sequence = tool_call_sequence(result["messages"])

    print("=== Final Answer ===")
    print(final)
    print("=== Tool Call Sequence ===")
    print(sequence)

    # 1. skin_analyze 호출 여부
    assert "skin_analyze" in sequence, "skin_analyze가 호출되지 않았습니다."

    # 2. search_pubmed 최소 1회 호출
    assert sequence.count("search_pubmed") >= 1, "search_pubmed가 호출되지 않았습니다."

    # 3. Final Answer에 논문 출처(연도 또는 et al.)
    assert any(token in final for token in ["et al.", "20", "PMID", "pmid"]), \
        "Final Answer에 논문 출처가 보이지 않습니다."

    # 4. '전문의' 문구
    assert "전문의" in final, "Final Answer에 '전문의' 문구가 없습니다."

    # 5. '반드시' 단정 표현 없음
    assert "반드시" not in final, "Final Answer에 '반드시' 단정 표현이 포함되어 있습니다."

    # 6. skin_analyze가 search_pubmed보다 먼저 호출
    assert sequence.index("skin_analyze") < sequence.index("search_pubmed"), \
        "skin_analyze가 search_pubmed보다 늦게 호출되었습니다."

    print("\n[PASS] 시나리오 1 모든 검증 통과")


if __name__ == "__main__":
    test_scenario_1()
