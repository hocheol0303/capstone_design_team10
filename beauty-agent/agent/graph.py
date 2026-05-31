"""Beauty Agent — LangGraph StateGraph 빌더와 ChatSession wrapper.

그래프 구조 (2단계 계층 라우팅):
    START → classify_intent
              ├─ think → conditional(act | finish)
              │   ↑                                          │
              │   │ should_continue: think|inject_pubmed|fin │
              │  observe ← act ← inject_pubmed ◄─────────────┘
              │           ▲___________│ (추천 직후 근거 검색 1회 강제)
              │
              └─ data_gate
                    ├─ compress → final_report → END
                    └─ insufficient_response → END

inject_recommend: 진단(skin_scores) 후 추천 의도인데 db_recommendations가 없으면
LLM 판단과 무관하게 recommend_treatment_db를 1회 강제(db_forced 가드). 시술명 환각 방지.
inject_pubmed: 시술 추천(db_recommendations)이 나왔는데 PubMed 근거가 없으면
LLM 판단과 무관하게 search_pubmed를 1회 강제 호출(pubmed_forced 가드로 1회만).

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
    classify_intent_node,
    compress,
    data_gate,
    final_report,
    finish,
    inject_pubmed_call,
    inject_recommend_call,
    insufficient_response,
    observe,
    think,
)
from agent.routers import (
    route_after_classify,
    route_after_data_gate,
    route_after_think,
    should_continue,
)
from agent.state import BeautyAgentState


def build_graph(checkpointer: InMemorySaver | None = None):
    """2단계 계층 라우팅 + ReAct 사이클 + 레포트 작성 경로.

    1) classify_intent: 사용자 메시지의 의도(report | general)를 판정.
    2) route_after_classify: report면 data_gate로, 아니면 think로.
    3) data_gate(report 한정): 데이터 충분 시 compress→final_report, 부족 시 insufficient_response.
    """
    workflow = StateGraph(BeautyAgentState)
    workflow.add_node("classify_intent",       classify_intent_node)
    workflow.add_node("data_gate",             data_gate)
    workflow.add_node("think",                 think)
    workflow.add_node("act",                   act)
    workflow.add_node("observe",               observe)
    workflow.add_node("inject_recommend",      inject_recommend_call)
    workflow.add_node("inject_pubmed",         inject_pubmed_call)
    workflow.add_node("finish",                finish)
    workflow.add_node("compress",              compress)
    workflow.add_node("final_report",          final_report)
    workflow.add_node("insufficient_response", insufficient_response)

    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges("classify_intent", route_after_classify, {
        "think":     "think",
        "data_gate": "data_gate",
    })
    workflow.add_conditional_edges("data_gate", route_after_data_gate, {
        "compress":              "compress",
        "insufficient_response": "insufficient_response",
    })

    workflow.add_conditional_edges("think", route_after_think, {
        "act":    "act",
        "finish": "finish",
    })
    workflow.add_edge("act", "observe")
    workflow.add_conditional_edges("observe", should_continue, {
        "think":            "think",
        "finish":           "finish",
        "inject_recommend": "inject_recommend",
        "inject_pubmed":    "inject_pubmed",
    })
    workflow.add_edge("inject_recommend", "act")
    workflow.add_edge("inject_pubmed", "act")

    workflow.add_edge("compress", "final_report")
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
        # ReAct 루프 가드는 "한 사용자 턴" 단위여야 한다. iteration_count/pubmed_forced를
        # 매 입력마다 리셋하지 않으면 세션이 길어질수록 max_iterations에 조기 도달해
        # (강제 search_pubmed 직후의) 최종 종합 답변이 잘려나간다.
        updates: dict[str, Any] = {
            "messages": [HumanMessage(content=user_text)],
            "iteration_count": 0,
            "db_forced": False,
            "pubmed_forced": False,
        }
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
                    # classify_intent/compress 등 내부 LLM 단계의 토큰은 사용자에게 노출하지 않는다.
                    if node_name not in ("think", "final_report"):
                        continue
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
