"""Phase 1 에이전트 그래프.

- LangChain 1.x의 `langchain.agents.create_agent`로 구성(내부는 LangGraph 그래프).
- 도구는 skin_analyze 단일 → 향후 단계에서 추가.
- 대화형 사용을 위해 ChatSession을 함께 제공한다(이전 메시지를 누적해 다음 invoke에 넘김).
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from agent.prompts import SYSTEM_PROMPT
from config import AI_MODEL
from tools.skin_analyze import skin_analyze


def build_agent():
    llm = init_chat_model(
        model=AI_MODEL,
        temperature=0,
    )
    tools = [skin_analyze]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def _extract_text(content) -> str:
    """모델별 응답 content를 사람이 읽기 좋은 문자열로 정규화.
    Gemini 등은 content를 [{type:'text', text:'...'}, ...] 리스트로 돌려준다.
    """
    
    # content가 이미 문자열이면 그대로 반환, 아니면 문자열까지 다 꺼내서 반환
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content)


class ChatSession:
    """대화 컨텍스트를 유지하는 단순 세션.

    create_agent로 만든 그래프는 invoke 단위로 동작하므로,
    여러 턴에 걸친 대화 맥락을 유지하려면 누적된 messages를 다시 넣어줘야 한다.
    """

    def __init__(self):
        self.agent = build_agent()
        self.messages: list = []

    def send(self, user_text: str) -> str:
        self.messages.append(HumanMessage(content=user_text))
        result = self.agent.invoke({"messages": self.messages})
        self.messages = result["messages"]
        return _extract_text(self.messages[-1].content)


def run_agent_once(user_text: str) -> dict:
    """단일 invoke용 헬퍼 (테스트/스크립트용)."""
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=user_text)]})
    last = result["messages"][-1]
    return {
        "final_answer": _extract_text(last.content),
        "messages":     result["messages"],
    }
