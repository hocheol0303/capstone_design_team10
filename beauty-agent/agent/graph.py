"""Phase 1 에이전트 그래프.

- LangChain 1.x의 `langchain.agents.create_agent`로 구성(내부는 LangGraph 그래프).
- 도구는 skin_analyze 단일 → 향후 단계에서 추가.
- 채팅 인터페이스 전용. 토큰 단위 스트리밍 출력을 ChatSession.stream으로 제공.
- 상태 영속화는 InMemorySaver + thread_id로 자동 처리.
"""
import uuid
from typing import Iterator

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.prompts import SYSTEM_PROMPT
from agent.state import BeautyAgentState
from config import AI_MODEL
from tools.recommend_treatment import recommend_treatment
from tools.skin_analyze import skin_analyze


def build_agent(checkpointer=None):
    """Phase 1 에이전트 컴파일.

    checkpointer를 넘기면 그래프가 thread_id별로 state(messages + 커스텀 필드)를
    자동 persist한다. invoke 시에는 새 메시지만 보내면 된다.
    """
    llm = init_chat_model(
        model=AI_MODEL,
        temperature=0,
    )
    tools = [skin_analyze, recommend_treatment]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=BeautyAgentState,
        checkpointer=checkpointer,
    )


def _extract_text(content) -> str:
    """모델별 응답 content를 사람이 읽기 좋은 문자열로 정규화.
    Gemini 등은 content를 [{type:'text', text:'...'}, ...] 리스트로 돌려준다.
    """
    if content is None:
        return ""
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
        return "".join(parts)
    return str(content)


class ChatSession:
    """대화 컨텍스트를 유지하는 세션.

    LangGraph 체크포인터(InMemorySaver) + thread_id로 그래프가 state를 자동 persist.
    호출자는 매 turn 새 메시지만 보내면 된다(누적 messages 재투입 불필요).
    """

    def __init__(self, thread_id: str | None = None):
        self.checkpointer = InMemorySaver()
        self.agent = build_agent(checkpointer=self.checkpointer)
        self.thread_id = thread_id or str(uuid.uuid4())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def _snapshot(self) -> dict:
        return self.agent.get_state(self.config).values

    @property
    def messages(self) -> list:
        return self._snapshot().get("messages", [])

    @property
    def skin_scores(self):
        return self._snapshot().get("skin_scores")

    @property
    def top_concerns(self):
        return self._snapshot().get("top_concerns")

    @property
    def db_recommendations(self):
        return self._snapshot().get("db_recommendations")

    def stream(self, user_text: str) -> Iterator[str]:
        """LLM 토큰을 발생 즉시 yield. 같은 메시지 안의 빈 chunk(tool_call만 있는 경우)는 건너뜀."""
        for chunk, _meta in self.agent.stream(
            {"messages": [HumanMessage(content=user_text)]},
            config=self.config,
            stream_mode="messages",
        ):
            text = _extract_text(getattr(chunk, "content", None))
            if text:
                yield text

    def send(self, user_text: str) -> str:
        """stream을 모두 모아 최종 문자열로 반환(논스트리밍 사용처용)."""
        return "".join(self.stream(user_text))
