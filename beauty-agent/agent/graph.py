"""
Beauty Agent — 책 정석 ReAct StateGraph (think → act → observe → finish).

위키독스 LangGraph part 2-1-2의 패턴 그대로:
    START → think → conditional(act | finish)
            think                        ↑
              │                          │
              ▼                          │
             act (ToolNode) ─→ observe ──┤ should_continue: think | finish
                                         ▼
                                       finish ─→ END

- think: LLM이 SystemMessage + history를 보고 thought 본문 + (필요 시) tool_calls 발행
- act:   LangGraph의 prebuilt ToolNode가 last AIMessage.tool_calls를 실행
         (도구별 가드는 각 도구 함수 본문에서 직접 수행)
- observe: 가장 최근 ToolMessage를 observations 리스트에 적재
- finish:  최종 답변(content)을 final_answer에 적재하고 is_complete=True

루프 가드: iteration_count >= max_iterations이면 강제 finish.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Iterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.state import BeautyAgentState
from config import AI_MODEL
from tools.recommend_treatment_db import recommend_treatment_db as recommend_treatment_db_tool
from tools.search_pubmed import search_pubmed as search_pubmed_tool
from tools.skin_analyze import skin_analyze as skin_analyze_tool

DEFAULT_MAX_ITERATIONS = 6


# ─────────────────────────── 헬퍼 ───────────────────────────

# LLM 모델 가져오기
def _load_llm():
    return init_chat_model(model=AI_MODEL, temperature=0)


# 들어온 messages 중에서 가장 최근 거가 사용자 입력이면 그 텍스트를 반환, 아예 없으면 빈 텍스트 반환
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


# 사용자 메시지의 '최종 레포트 작성' 의도 여부를 LLM으로 분류
_INTENT_SYSTEM = (
    "당신은 한국어 사용자 메시지를 두 카테고리로 분류하는 의도 분류기입니다. "
    "정확히 'report' 또는 'general' 한 단어로만 답변하세요.\n"
    "- report: 누적된 진단/추천 데이터를 종합한 최종 보고서·요약·정리·총평을 요청하는 의도. "
    "예) '최종 레포트 작성해줘', '지금까지 내용 정리해줘', '총정리 부탁', '요약 보고서 만들어줘'.\n"
    "- general: 그 외 모든 의도. 예) 진단 요청, 시술 추천 요청, 일반 대화, 후속 질문 등."
)


def _classify_intent(user_text: str) -> str:
    """
    LLM 기반 의도 분류. 반환: 'report' 또는 'general'.

    호출 실패 또는 입력이 비어 있으면 안전하게 'general' 반환.
    """
    if not user_text or not user_text.strip():
        return "general"
    try:
        llm = _load_llm()
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=f"사용자 메시지: {user_text}\n분류:"),
        ])
        label = _extract_text(getattr(response, "content", "")).strip().lower()
        return "report" if "report" in label else "general"
    except Exception:  # noqa: BLE001
        return "general"

# 들어온 content에서 text 부분만 추출하는 함수
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


def _build_system_context(state: BeautyAgentState) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "현재 state 요약:\n"
        f"- image_path: {state.get('image_path') or 'None'}\n"
        f"- gender: {state.get('gender') or 'None'}\n"
        f"- skin_scores: {'존재' if state.get('skin_scores') else '없음'}\n"
        f"- db_recommendations: {'존재' if state.get('db_recommendations') else '없음'}\n"
        f"- pubmed_recommendations: {'존재' if state.get('pubmed_recommendations') else '없음'}\n"
        f"- iteration: {state.get('iteration_count', 0)}/{state.get('max_iterations', DEFAULT_MAX_ITERATIONS)}\n"
    )


def _extract_state_context(state: BeautyAgentState) -> dict[str, str | None]:
    """
    가장 최근 HumanMessage(text)에서 image_path/gender를 파싱해 state에 정착시킨다.
    """
    text = _latest_human_message_text(state.get("messages") or [])
    image_path = _parse_image_path(text)
    gender = _parse_gender(text)
    updates: dict[str, str | None] = {}
    if image_path:
        updates["image_path"] = image_path
    if gender:
        updates["gender"] = gender
    return updates


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
    return "입력에 처리할 수 없는 문자가 포함되어 있습니다. 다시 입력해 주세요."


# ─────────────────────────── 노드 ───────────────────────────

def think(state: BeautyAgentState) -> dict[str, Any]:
    """
    LLM이 SystemMessage + history를 보고 다음 행동을 결정.

    출력:
      - messages: [AIMessage]            (LangGraph가 add_messages로 머지)
      - thoughts: [텍스트]               (content text. 비어 있어도 OK)
      - actions:  [tool_call dict]       (tool_call이 있을 때만)
      - iteration_count: +1
      - current_goal:    초기 1회 세팅
      - max_iterations:  미설정 시 기본값
      - image_path/gender: 사용자 입력에서 파싱한 컨텍스트
    """
    context_updates = _extract_state_context(state)     # 가장 최근 사용자 입력에서 image_path, gender를 추출해서 state 업데이트에 포함시킨다.
    merged_state = {**state, **context_updates}         # state는 불변이므로 merged_state라는 새 dict에 병합해서 이후 처리에서 사용한다.

    # chat 모델이 호출할 수 있는 도구들을 정의
    llm = _load_llm().bind_tools(
        [skin_analyze_tool, recommend_treatment_db_tool, search_pubmed_tool]
    )

    # 시스템 프롬프트 + state 요약을 system message로 만들어서 LLM에 전달한다.
    messages = [SystemMessage(content=_build_system_context(merged_state))]
    messages.extend(merged_state.get("messages") or [])
    response = llm.invoke(messages)

    # response에서 thought_text와 action_dict를 추출한다.
    thought_text = _extract_text(getattr(response, "content", None))
    action_dict = response.tool_calls[0] if getattr(response, "tool_calls", None) else None

    updates: dict[str, Any] = {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    # 업데이트할 state가 있으면 추가한다.
    if thought_text:
        updates["thoughts"] = [thought_text]
    if action_dict:
        updates["actions"] = [dict(action_dict)]

    # 초기화 (한 번만 세팅)
    if not state.get("current_goal"):
        goal = _latest_human_message_text(state.get("messages") or [])
        if goal:
            updates["current_goal"] = goal[:200]
    if not state.get("max_iterations"):
        updates["max_iterations"] = DEFAULT_MAX_ITERATIONS

    updates.update(context_updates)
    return updates


# act = 책 표준 ToolNode.
# 도구별 가드(image_path 필수, skin_scores 필수 등)는 각 도구 함수 본문에서 직접 수행.
act = ToolNode([skin_analyze_tool, recommend_treatment_db_tool, search_pubmed_tool])


def observe(state: BeautyAgentState) -> dict[str, Any]:
    """
    가장 최근 ToolMessage들을 observations 리스트에 적재.

    ToolNode는 직전 AIMessage의 tool_calls 개수만큼 ToolMessage를 append한다.
    여기서는 messages 끝에서부터 거꾸로 ToolMessage들을 모은 뒤(다음 AIMessage를 만나면 중단)
    각 content를 observations에 push.
    """
    messages = state.get("messages") or []
    collected: list[str] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            text = _extract_text(msg.content)
            if text:
                collected.append(text)
        elif isinstance(msg, AIMessage):
            break
    if not collected:
        return {}
    # 시간 순서대로 (오래된 → 최근)
    collected.reverse()
    return {"observations": collected}


def finish(state: BeautyAgentState) -> dict[str, Any]:
    """
    최종 답변 content를 final_answer로 발췌하고 is_complete=True.
    """
    last_ai_text = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            last_ai_text = _extract_text(getattr(msg, "content", "")) or last_ai_text
            if last_ai_text:
                break
    return {
        "final_answer": last_ai_text,
        "is_complete": True,
    }


def final_report(state: BeautyAgentState) -> dict[str, Any]:
    """
    누적 state(skin_scores, db_recommendations, pubmed_recommendations)를 종합해
    최종 환자용 레포트를 작성하고 대화를 종료한다.

    도구는 호출하지 않으며 LLM이 텍스트 답변만 생성한다. is_complete=True.
    누적 자료가 비어 있으면 그 사실을 명시적으로 안내한다.
    """
    skin_scores = state.get("skin_scores") or {}
    db_recs = state.get("db_recommendations") or []
    pubmed_recs = state.get("pubmed_recommendations") or []
    history_text = _latest_human_message_text(state.get("messages") or [])

    parts: list[str] = ["다음 누적 자료를 바탕으로 환자용 최종 레포트를 작성하세요."]
    if skin_scores.get("raw_scores"):
        parts.append(
            "\n[피부 진단 결과]\n"
            f"- 추정 연령: {skin_scores.get('age')}\n"
            f"- 성별: {skin_scores.get('gender_input') or state.get('gender')}\n"
            f"- 부위별 raw 점수(0~100, 낮을수록 심각): {skin_scores['raw_scores']}"
        )
    if db_recs:
        parts.append(
            f"\n[DB 추천 시술 ({len(db_recs)}건)]\n" + "\n".join(
                f"- {r.get('region_ko')} ({r.get('score')}점) → code {r.get('code')}: {r.get('treatment')}"
                + (f"\n  안내: {(r.get('customer_desc') or '').strip()}" if r.get('customer_desc') else "")
                for r in db_recs
            )
        )
    if pubmed_recs:
        parts.append(
            f"\n[PubMed 논문 근거 ({len(pubmed_recs)}건)]\n" + "\n".join(
                f"- {r.get('region_ko')} ({r.get('score')}점, {r.get('severity')}): "
                f"{r.get('title')} — {r.get('authors')} {r.get('year')} (PMID {r.get('pmid')})"
                for r in pubmed_recs
            )
        )
    if len(parts) == 1:
        parts.append(
            "\n현재 누적된 진단/추천/근거 자료가 없습니다. "
            "사용자에게 자료 없음을 솔직히 안내하고, 진단(skin_analyze)부터 진행할 것을 권하세요."
        )
    if history_text:
        parts.append(f"\n[사용자의 직전 요청]\n{history_text}")

    sys_prompt = (
        "당신은 피부과 코디네이터입니다. 누적된 자료를 종합해 환자용 최종 레포트를 작성하세요.\n"
        "구성: ① 진단 요약 ② 권장 시술 (DB 매칭 + 코드) ③ 시술의 학술적 근거 (PubMed: 저자/연도/PMID 인용) ④ 종합 조언.\n"
        "도구는 호출하지 마세요. 의학적 단정 표현은 피하고 '~로 보입니다' 형태를 사용하세요.\n"
        "전체 분량은 사용자가 한눈에 읽을 수 있도록 markdown으로 깔끔하게 정리합니다."
    )
    llm = _load_llm()  # bind_tools 안 함 — 순수 텍스트
    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content="\n".join(parts)),
    ])

    return {
        "messages":     [response],
        "final_answer": _extract_text(getattr(response, "content", "")),
        "is_complete":  True,
    }

def data_check(state: BeautyAgentState) -> dict:
    """
    edge 만들기 위한 임시 노드
    아무런 업데이트 없이 state를 그대로 반환하지만, think에서 이 노드로 분기할 때 진단/추천 데이터 존재 여부에 따라 분기하도록 라우팅 함수에서 체크한다.
    """
    return {}

def insufficient_response(state) -> dict:
    msg = AIMessage(content=(
        "최종 레포트를 작성하려면 진단 결과와 시술 추천/논문 근거가 누적되어야 합니다.\n"
        f"- 진단(skin_analyze): {'✅' if state.get('skin_scores') else '❌'}\n"
        f"- 시술 추천(recommend_treatment_db): {'✅' if state.get('db_recommendations') else '❌'}\n"
        f"- 논문 근거(search_pubmed): {'✅' if state.get('pubmed_recommendations') else '❌'}\n"
        "먼저 위 항목을 진행해 주세요."
    ))
    text = _extract_text(msg.content)
    return {"messages": [msg], "final_answer": text, "is_complete": True}



# ─────────────────────────── 라우팅 ───────────────────────────

def route_after_think(state: BeautyAgentState) -> str:
    """
    think이 tool_call을 발행했으면 act, 아니면 곧장 finish.

    이 분기가 책 패턴의 'add_edge(think, act)' 부분을 약간 풀어 쓴 형태.
    ToolNode가 tool_call 없는 메시지에서 에러를 낼 수 있어 안전 분기.
    """
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "act"
    return "finish"


def should_continue(state: BeautyAgentState) -> str:
    """
    observe 이후: 루프 계속 / 종료 결정.

    - iteration_count가 max_iterations에 도달했으면 finish (loop guard).
    - 그 외에는 think로 돌아가 LLM이 추가 행동 또는 최종 답변을 결정.
    """
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if iter_count >= max_iter:
        return "finish"
    return "think"


def route_from_start(state: BeautyAgentState) -> str:
    """
    첫 진입 분기. LLM이 사용자 메시지를 보고 '최종 레포트 요청' 의도로 분류하면
    think 건너뛰고 곧장 final_report. 그 외에는 일반 ReAct 사이클(think) 시작.
    """
    latest = _latest_human_message_text(state.get("messages") or [])

    # 사용자가 레포트를 작성해달라고 했는지를 판단함 -> 데이터가 충분한지 판단
    if _classify_intent(latest) == "report":
        return "data_check"
    return "think"

def route_after_data_check(state: BeautyAgentState) -> str:
    has_diag = bool((state.get("skin_scores") or {}).get("raw_scores"))
    has_db = bool(state.get("db_recommendations"))
    has_pub = bool(state.get("pubmed_recommendations"))

    if not has_diag or not (has_db or has_pub):
        return "incomplete"
    else:
        return "complete"

# ─────────────────────────── 그래프 ───────────────────────────

def build_graph(checkpointer: InMemorySaver | None = None):
    """think → act → observe → conditional(think|finish) → END 사이클 + final_report 우회.

    START에서 사용자 메시지가 '레포트' 트리거를 포함하면 think 건너뛰고 final_report로 직행.
    """
    workflow = StateGraph(BeautyAgentState)
    workflow.add_node("think",        think)
    workflow.add_node("act",          act)
    workflow.add_node("observe",      observe)
    workflow.add_node("finish",       finish)
    workflow.add_node("final_report", final_report)
    workflow.add_node("data_check", data_check)
    workflow.add_node("insufficient_response", insufficient_response)

    workflow.add_conditional_edges(START, route_from_start, {
        "think":        "think",
        "data_check": "data_check"
    })

    workflow.add_conditional_edges("data_check", route_after_data_check,{
        "complete": "final_report",
        "incomplete": "insufficient_response"
    })

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
        parsed_image_path = _parse_image_path(user_text)
        parsed_gender = _parse_gender(user_text)
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
                    text = _extract_text(getattr(chunk, "content", None))
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
                                    args_str = _format_tool_args(call.get("args"))
                                    yield f"\n\n🔧 [Act] {call.get('name')}({args_str})\n"
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
    "act",
    "build_graph",
    "final_report",
    "finish",
    "observe",
    "route_after_think",
    "route_from_start",
    "should_continue",
    "think",
]
