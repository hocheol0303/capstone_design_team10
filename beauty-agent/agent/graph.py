"""Beauty Agent — LangGraph StateGraph 빌더와 ChatSession wrapper.

그래프 구조:
    START → route_from_start
              ├─ think → conditional(act | finish)
              │   ↑                              │
              │   │ should_continue: think|finish│
              │  observe ← act ←─────────────────┘
              │
              ├─ final_report → END
              └─ insufficient_response → END

노드 정의는 [nodes.py](nodes.py), 라우팅은 [routers.py](routers.py),
헬퍼는 [helpers.py](helpers.py)에 분리되어 있다.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.helpers import (
    extract_text,
    format_tool_args,
    parse_gender,
    parse_image_path,
    reinput_request_message,
)
from agent.nodes import (
    act,
    final_report,
    finish,
    insufficient_response,
    observe,
    think,
)
from agent.routers import route_after_think, route_from_start, should_continue
from agent.state import BeautyAgentState


def build_graph(checkpointer: InMemorySaver | None = None):
    """think → act → observe → conditional(think|finish) → END 사이클 + 레포트 우회 분기.

    START에서 사용자 메시지가 '레포트' 트리거로 분류되면 think를 건너뛰고
    데이터 충분 시 final_report, 부족 시 insufficient_response로 분기한다.
    """
    workflow = StateGraph(BeautyAgentState)
    workflow.add_node("think",        think)
    workflow.add_node("act",          act)
    workflow.add_node("observe",      observe)
    workflow.add_node("finish",       finish)
    workflow.add_node("final_report", final_report)
    workflow.add_node("insufficient_response", insufficient_response)

    workflow.add_conditional_edges(START, route_from_start)

    workflow.add_conditional_edges("think", route_after_think, {
        "act":    "act",
        "finish": "finish",
    })
    workflow.add_edge("act", "observe")
    workflow.add_conditional_edges("observe", should_continue, {
        "think":  "think",
        "finish": "finish",
    })

    workflow.add_edge("insufficient_response", END)
    workflow.add_edge("finish",       END)
    workflow.add_edge("final_report", END)

    return workflow.compile(checkpointer=checkpointer)


# ─────────────────────────── ChatSession ───────────────────────────

class ChatSession:
    """thread_id별 영속화된 LangGraph 세션 wrapper.

    stream()은 messages+updates 혼합 스트림을 받아 사용자에게 다음 형태로 노출한다:
      - 🧠 [Thought] : think 노드의 LLM 텍스트 (사이클당 1회 prefix)
      - 🔧 [Act]     : think이 tool_call을 발행한 직후 한 줄
      - 📋 [Observe] : act(ToolNode)가 ToolMessage를 produce한 직후 블록
      - 📑 [Final Report] : 사용자가 '레포트' 트리거를 입력해 final_report 노드가 작성한 답변
    """

    def __init__(self, thread_id: str | None = None):
        self.checkpointer = InMemorySaver()
        self.graph = build_graph(checkpointer=self.checkpointer)
        self.thread_id = thread_id or str(uuid.uuid4())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def _snapshot(self) -> dict:
        return self.graph.get_state(self.config).values

    # ── 책 정석 ReAct 노출 ──
    @property
    def messages(self) -> list:
        return self._snapshot().get("messages", [])

    @property
    def thoughts(self) -> list[str]:
        return self._snapshot().get("thoughts", [])

    @property
    def actions(self) -> list[dict]:
        return self._snapshot().get("actions", [])

    @property
    def observations(self) -> list[str]:
        return self._snapshot().get("observations", [])

    @property
    def iteration_count(self) -> int:
        return self._snapshot().get("iteration_count", 0)

    @property
    def final_answer(self) -> str:
        return self._snapshot().get("final_answer", "")

    @property
    def is_complete(self) -> bool:
        return bool(self._snapshot().get("is_complete"))

    # ── 도메인 노출 ──
    @property
    def skin_scores(self):
        return self._snapshot().get("skin_scores")

    @property
    def top_concerns(self):
        return self._snapshot().get("top_concerns")

    @property
    def db_recommendations(self):
        return self._snapshot().get("db_recommendations")

    @property
    def pubmed_recommendations(self):
        return self._snapshot().get("pubmed_recommendations")

    # ── 입력/스트림 ──

    def _initial_state(self, user_text: str) -> dict[str, Any]:
        updates: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
        parsed_image_path = parse_image_path(user_text)
        parsed_gender = parse_gender(user_text)
        if parsed_image_path:
            updates["image_path"] = parsed_image_path
        if parsed_gender:
            updates["gender"] = parsed_gender
        return updates

    def stream(self, user_text: str) -> Iterator[str]:
        """ReAct 단계를 순차적으로 yield.

        - messages mode → think 노드의 LLM 텍스트(Thought/Final) prefix 없이 흘림.
        - updates mode → AIMessage.tool_calls는 🔧 [Act],
                          ToolMessage는 📋 [Observe] 블록으로 변환.
        """
        sent_tool_calls: set = set()
        sent_tool_msgs: set = set()
        thought_prefixed = False
        report_prefixed = False

        try:
            for kind, payload in self.graph.stream(
                self._initial_state(user_text),
                config=self.config,
                stream_mode=["messages", "updates"],
            ):
                if kind == "messages":
                    chunk, meta = payload
                    if isinstance(chunk, ToolMessage):
                        continue  # updates에서 한 블록으로 처리
                    node_name = (meta or {}).get("langgraph_node") if isinstance(meta, dict) else None
                    text = extract_text(getattr(chunk, "content", None))
                    if text:
                        if node_name == "final_report":
                            if not report_prefixed:
                                yield "\n\n📑 [Final Report]\n"
                                report_prefixed = True
                            yield text
                        else:
                            if not thought_prefixed:
                                yield "🧠 [Thought] "
                                thought_prefixed = True
                            yield text
                elif kind == "updates":
                    for _node_name, node_update in payload.items():
                        msgs = (node_update or {}).get("messages") or []
                        for msg in msgs:
                            if isinstance(msg, AIMessage):
                                for call in (msg.tool_calls or []):
                                    cid = call.get("id") or f"_{call.get('name')}"
                                    if cid in sent_tool_calls:
                                        continue
                                    sent_tool_calls.add(cid)
                                    args_str = format_tool_args(call.get("args"))
                                    yield f"\n\n🔧 [Act] {call.get('name')}({args_str})\n"
                                    thought_prefixed = False
                            elif isinstance(msg, ToolMessage):
                                key = (msg.tool_call_id, id(msg))
                                if key in sent_tool_msgs:
                                    continue
                                sent_tool_msgs.add(key)
                                content = extract_text(msg.content)
                                if content:
                                    yield f"\n📋 [Observe]\n{content}\n\n"
                                    thought_prefixed = False
        except UnicodeEncodeError:
            yield reinput_request_message()
        except Exception as exc:  # noqa: BLE001
            if "surrogates not allowed" in str(exc):
                yield reinput_request_message()
            else:
                raise

    def send(self, user_text: str) -> str:
        return "".join(self.stream(user_text))


__all__ = [
    "BeautyAgentState",
    "ChatSession",
    "build_graph",
]
