"""Beauty Agent — 명시적 StateGraph 기반 ReAct 구현.

create_agent의 블랙박스 대신 LangGraph StateGraph를 직접 조립한다.

흐름:
    agent_node ─┬─ tool_calls[0].name == 'skin_analyze'          → skin_analyze_node ─┐
                ├─ tool_calls[0].name == 'recommend_treatment_db' → recommendation_node ┤
                └─ (tool_call 없음)                                → END               │
                                                                                       ↓
                                                       ←─────────────────  agent_node

- agent_node: LLM이 SystemMessage + 대화 history를 보고 다음 tool 또는 최종 답변 결정.
- skin_analyze_node, recommendation_node: 실제 [tools/](../tools) 의 @tool 함수를 호출하고
  Command(update={...})를 받아 state에 반영.
- ChatSession.stream은 messages+updates 혼합 스트림을 받아 Thought / Act / Observe를
  순차적으로 사용자에게 노출한다.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Iterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agent.prompts import SYSTEM_PROMPT
from agent.state import BeautyAgentState
from config import AI_MODEL
from tools.recommend_treatment_db import recommend_treatment_db as recommend_treatment_db_tool
from tools.skin_analyze import skin_analyze as skin_analyze_tool


# ─────────────────────────── 헬퍼 ───────────────────────────

def _invoke_tool(tool_obj, /, **kwargs):
    """LangChain @tool로 감싼 StructuredTool은 직접 호출이 안 되어 `.func`(raw)로 호출.

    InjectedToolCallId/InjectedState 인자는 일반 kwargs로 통과시킨다.
    """
    func = getattr(tool_obj, "func", None)
    if callable(func):
        return func(**kwargs)
    # raw function이 들어온 경우의 폴백
    if callable(tool_obj):
        return tool_obj(**kwargs)
    for name in ("invoke", "run"):
        meth = getattr(tool_obj, name, None)
        if callable(meth):
            return meth(**kwargs)
    raise RuntimeError("Cannot invoke tool object: unsupported tool type")


def _load_llm():
    return init_chat_model(model=AI_MODEL, temperature=0)


def _latest_human_message_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _parse_image_path(text: str) -> str | None:
    patterns = [
        r"(?:image_path|path|경로)\s*[:=]\s*([^\s,]+)",
        r"(?:image|img)\s*[:=]\s*([^\s,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("'\"")
    inline_path = re.search(r"([^\s]+\.(?:jpg|jpeg|png|webp|bmp|gif))", text, re.IGNORECASE)
    if inline_path:
        return inline_path.group(1).strip().strip("'\"")
    return None


def _parse_gender(text: str) -> str | None:
    normalized = text.lower()
    if re.search(r"(?:\b여자\b|\bfemale\b|\bwoman\b)", normalized):
        return "female"
    if re.search(r"(?:\b남자\b|\bmale\b|\bman\b)", normalized):
        return "male"
    return None


def _first_tool_call(state: BeautyAgentState) -> dict[str, Any] | None:
    messages = state.get("messages") or []
    if not messages:
        return None
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return last_message.tool_calls[0]
    return None


def _build_system_context(state: BeautyAgentState) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "현재 state 요약:\n"
        f"- image_path: {state.get('image_path') or 'None'}\n"
        f"- gender: {state.get('gender') or 'None'}\n"
        f"- skin_scores: {'존재' if state.get('skin_scores') else '없음'}\n"
        f"- db_recommendations: {'존재' if state.get('db_recommendations') else '없음'}\n"
    )


def _extract_state_context(state: BeautyAgentState) -> dict[str, str | None]:
    """가장 최근 HumanMessage에서 image_path/gender를 파싱해 state에 정착시킨다."""
    text = _latest_human_message_text(state.get("messages") or [])
    image_path = _parse_image_path(text)
    gender = _parse_gender(text)
    updates: dict[str, str | None] = {}
    if image_path:
        updates["image_path"] = image_path
    if gender:
        updates["gender"] = gender
    return updates


# ─────────────────────────── 노드 ───────────────────────────

def agent_node(state: BeautyAgentState) -> dict[str, Any]:
    """LLM을 bind_tools해서 다음 tool 호출 또는 최종 답변을 생성한다."""
    llm = _load_llm().bind_tools([skin_analyze_tool, recommend_treatment_db_tool])
    context_updates = _extract_state_context(state)
    merged_state = {**state, **context_updates}

    messages = [SystemMessage(content=_build_system_context(merged_state))]
    messages.extend(merged_state.get("messages") or [])

    response = llm.invoke(messages)

    updates: dict[str, Any] = {"messages": [response]}
    updates.update(context_updates)
    return updates


def skin_analyze_node(state: BeautyAgentState) -> dict[str, Any]:
    """tools.skin_analyze 도구를 호출해 raw 점수 + top_concerns를 state에 반영."""
    tool_call = _first_tool_call(state) or {}
    tool_call_id = str(tool_call.get("id", ""))

    if state.get("skin_scores") is not None:
        message = ToolMessage(
            content="skin_scores already exist. Skipping redundant skin analysis.",
            tool_call_id=tool_call_id,
        )
        return {
            "messages": [message],
            "reasoning_steps": ["skin_analyze skipped: skin_scores already in state."],
        }

    image_path = tool_call.get("args", {}).get("image_path") or state.get("image_path")
    gender = tool_call.get("args", {}).get("gender") or state.get("gender")

    if not image_path:
        message = ToolMessage(
            content="Error: image_path is required before skin analysis.",
            tool_call_id=tool_call_id,
        )
        return {
            "messages": [message],
            "reasoning_steps": ["skin_analyze failed: image_path missing."],
        }

    try:
        cmd = _invoke_tool(
            skin_analyze_tool,
            image_path=image_path,
            gender=gender,
            tool_call_id=tool_call_id,
        )
    except Exception as exc:  # noqa: BLE001
        msg = ToolMessage(content=f"Error: skin_analyze tool raised: {exc}", tool_call_id=tool_call_id)
        return {"messages": [msg], "reasoning_steps": [f"skin_analyze failed: {exc}"]}

    update = getattr(cmd, "update", cmd) or {}
    update.setdefault("image_path", image_path)
    update.setdefault("gender", gender)
    steps = update.get("reasoning_steps") or []
    steps.append(f"skin_analyze completed for {image_path}.")
    update["reasoning_steps"] = steps
    return update


def recommendation_node(state: BeautyAgentState) -> dict[str, Any]:
    """tools.recommend_treatment_db 도구를 호출해 AuraDB 추천을 state에 반영."""
    tool_call = _first_tool_call(state) or {}
    tool_call_id = str(tool_call.get("id", ""))

    if state.get("skin_scores") is None:
        message = ToolMessage(
            content="Error: recommend_treatment_db requires skin_scores first. Call skin_analyze.",
            tool_call_id=tool_call_id,
        )
        return {
            "messages": [message],
            "reasoning_steps": ["recommend_treatment_db blocked: skin_scores missing."],
        }

    try:
        cmd = _invoke_tool(
            recommend_treatment_db_tool,
            state=state,
            tool_call_id=tool_call_id,
        )
    except Exception as exc:  # noqa: BLE001
        msg = ToolMessage(content=f"Error: recommend_treatment_db tool raised: {exc}", tool_call_id=tool_call_id)
        return {"messages": [msg], "reasoning_steps": [f"recommend_treatment_db failed: {exc}"]}

    update = getattr(cmd, "update", cmd) or {}
    steps = update.get("reasoning_steps") or []
    steps.append("recommend_treatment_db completed via auradb.")
    update["reasoning_steps"] = steps
    return update


def should_continue(state: BeautyAgentState) -> str:
    """last AIMessage의 첫 tool_call에 따라 다음 노드 결정."""
    tool_call = _first_tool_call(state)
    if tool_call is None:
        return END

    name = tool_call.get("name")
    if name == "skin_analyze":
        return "skin_analyze_node"
    if name == "recommend_treatment_db":
        return "recommendation_node"
    return END


def build_graph(checkpointer: InMemorySaver | None = None):
    """Beauty Agent StateGraph 컴파일."""
    workflow = StateGraph(BeautyAgentState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("skin_analyze_node", skin_analyze_node)
    workflow.add_node("recommendation_node", recommendation_node)

    workflow.set_entry_point("agent_node")
    workflow.add_conditional_edges("agent_node", should_continue)
    workflow.add_edge("skin_analyze_node", "agent_node")
    workflow.add_edge("recommendation_node", "agent_node")

    return workflow.compile(checkpointer=checkpointer)


# ─────────────────────────── 출력 정규화 ───────────────────────────

def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _format_tool_args(args: dict | None) -> str:
    if not args:
        return ""
    pieces = []
    for key, value in args.items():
        if isinstance(value, str):
            pieces.append(f"{key}={value!r}")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def _reinput_request_message() -> str:
    return (
        "💭 [Thought] 입력에 처리할 수 없는 문자가 포함된 것으로 보입니다.\n"
        "✅ [Final] 입력에 처리할 수 없는 문자가 포함되어 있습니다. 다시 입력해 주세요."
    )


# ─────────────────────────── ChatSession ───────────────────────────

class ChatSession:
    """thread_id별 영속화된 LangGraph 세션 wrapper.

    stream()은 messages+updates 혼합 스트림을 받아 다음 순서로 사용자에게 노출한다:
      💭 Thought : LLM이 streaming하는 reasoning/text 토큰
      🔧 Act     : agent_node가 tool_calls를 emit한 직후 (한 줄)
      📋 Observe : tool 노드가 ToolMessage를 emit한 직후 (블록)
      (Thought가 다시 이어지며) 최종 응답 → END
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

    @property
    def reasoning_steps(self):
        return self._snapshot().get("reasoning_steps", [])

    def _initial_state(self, user_text: str) -> dict[str, Any]:
        updates: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
        parsed_image_path = _parse_image_path(user_text)
        parsed_gender = _parse_gender(user_text)
        if parsed_image_path:
            updates["image_path"] = parsed_image_path
        if parsed_gender:
            updates["gender"] = parsed_gender
        return updates

    def stream(self, user_text: str) -> Iterator[str]:
        """ReAct 단계를 순차적으로 yield한다.

        - messages mode → AI 텍스트는 그대로(Thought/Final), ToolMessage는 무시(아래에서 처리).
        - updates mode → AIMessage.tool_calls는 🔧 [Act], ToolMessage는 📋 [Observe]로 변환.
        """
        sent_tool_calls: set = set()
        sent_tool_msgs: set = set()
        thought_prefixed = False

        def _ensure_thought_header() -> str:
            nonlocal thought_prefixed
            if thought_prefixed:
                return ""
            thought_prefixed = True
            return "💭 [Thought] "

        try:
            for kind, payload in self.graph.stream(
                self._initial_state(user_text),
                config=self.config,
                stream_mode=["messages", "updates"],
            ):
                if kind == "messages":
                    chunk, _meta = payload
                    if isinstance(chunk, ToolMessage):
                        continue  # updates에서 한 블록으로 처리
                    text = _extract_text(getattr(chunk, "content", None))
                    if text:
                        prefix = _ensure_thought_header()
                        yield prefix + text
                elif kind == "updates":
                    for _node_name, node_update in payload.items():
                        msgs = node_update.get("messages") or []
                        for msg in msgs:
                            if isinstance(msg, AIMessage):
                                for call in (msg.tool_calls or []):
                                    cid = call.get("id") or f"_{call.get('name')}"
                                    if cid in sent_tool_calls:
                                        continue
                                    sent_tool_calls.add(cid)
                                    args_str = _format_tool_args(call.get("args"))
                                    yield f"\n\n🔧 [Act] {call.get('name')}({args_str})\n"
                                    # 다음 LLM 응답에 다시 Thought 헤더 붙도록 리셋
                                    thought_prefixed = False
                            elif isinstance(msg, ToolMessage):
                                key = (msg.tool_call_id, id(msg))
                                if key in sent_tool_msgs:
                                    continue
                                sent_tool_msgs.add(key)
                                content = _extract_text(msg.content)
                                if content:
                                    yield f"\n📋 [Observe]\n{content}\n\n"
                                    thought_prefixed = False
        except UnicodeEncodeError:
            yield _reinput_request_message()
        except Exception as exc:  # noqa: BLE001
            if "surrogates not allowed" in str(exc):
                yield _reinput_request_message()
            else:
                raise

    def send(self, user_text: str) -> str:
        return "".join(self.stream(user_text))


__all__ = [
    "BeautyAgentState",
    "ChatSession",
    "agent_node",
    "build_graph",
    "recommendation_node",
    "should_continue",
    "skin_analyze_node",
]
