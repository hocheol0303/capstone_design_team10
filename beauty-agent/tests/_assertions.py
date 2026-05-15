"""테스트 공용 어설션 헬퍼."""
from langchain_core.messages import AIMessage, ToolMessage


def tool_call_sequence(messages):
    """messages에서 호출된 tool 이름을 호출 순서대로 반환."""
    names = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (getattr(m, "tool_calls", None) or []):
                names.append(tc["name"])
    return names


def tool_observations(messages, tool_name):
    """특정 tool의 ToolMessage 결과들을 반환."""
    return [m for m in messages if isinstance(m, ToolMessage) and m.name == tool_name]
