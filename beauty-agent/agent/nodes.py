"""Beauty agent LangGraph 노드 모음.

think → act → observe → finish / final_report / insufficient_response.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.helpers import (
    DEFAULT_MAX_ITERATIONS,
    build_system_context,
    classify_intent,
    extract_state_context,
    extract_text,
    latest_human_message_text,
    load_llm,
    severe_untreated_directive,
)
from agent.prompts import COMPRESS_PROMPT, FINAL_REPORT_PROMPT
from agent.state import BeautyAgentState
from config import AI_TEMPERATURE_CHAT
from tools.recommend_treatment_db import recommend_treatment_db as recommend_treatment_db_tool
from tools.search_pubmed import search_pubmed as search_pubmed_tool
from tools.skin_analyze import aggregate_regions
from tools.skin_analyze import skin_analyze as skin_analyze_tool


def classify_intent_node(state: BeautyAgentState) -> dict[str, Any]:
    """가장 최근 user 메시지의 의도를 분류해 state.intent에 저장.

    반환 값: 'report' | 'general'. 라우팅은 route_after_classify가 담당.
    """
    latest = latest_human_message_text(state.get("messages") or [])
    return {"intent": classify_intent(latest)}


def data_gate(state: BeautyAgentState) -> dict:
    """라우팅 전용 더미 노드.

    데이터 충분 여부는 route_after_data_gate가 체크해서
    compress 또는 insufficient_response로 분기한다.
    """
    return {}


def think(state: BeautyAgentState) -> dict[str, Any]:
    """LLM이 SystemMessage + history를 보고 다음 행동을 결정.

    출력:
      - messages: [AIMessage]            (LangGraph가 add_messages로 머지)
      - thoughts: [텍스트]               (content text. 비어 있어도 OK)
      - actions:  [tool_call dict]       (tool_call이 있을 때만)
      - iteration_count: +1
      - current_goal:    초기 1회 세팅
      - max_iterations:  미설정 시 기본값
      - image_path/gender: 사용자 입력에서 파싱한 컨텍스트
    """
    context_updates = extract_state_context(state)
    merged_state = {**state, **context_updates}

    # "LLM아, 너가 사용할 수 있는 도구를 알려줄게. 그리고 그 도구들의 설명은 docstring으로 제공될거야(@tool 데코레이터 사용된 노드 한정)."
    llm = load_llm(
        temperature=AI_TEMPERATURE_CHAT,
        frequency_penalty=0.3,
        max_tokens=2000,
    ).bind_tools(
        [skin_analyze_tool, recommend_treatment_db_tool, search_pubmed_tool]
    )

    messages = [SystemMessage(content=build_system_context(merged_state))]
    messages.extend(merged_state.get("messages") or [])
    response = llm.invoke(messages)

    thought_text = extract_text(getattr(response, "content", None))
    
    # act 노드(ToolNode)에서 state의 tool_calls를 보고 어떤 도구를 호출할지 결정함.
    # ToolNode가 state의 tool_calls 속성을 보고 도구 호출 여부와 호출할 도구를 결정함
    action_dict = response.tool_calls[0] if getattr(response, "tool_calls", None) else None

    updates: dict[str, Any] = {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    if thought_text:
        updates["thoughts"] = [thought_text]
    if action_dict:
        updates["actions"] = [dict(action_dict)]

    if not state.get("current_goal"):
        goal = latest_human_message_text(state.get("messages") or [])
        if goal:
            updates["current_goal"] = goal[:200]
    if not state.get("max_iterations"):
        updates["max_iterations"] = DEFAULT_MAX_ITERATIONS

    updates.update(context_updates)
    return updates


# act = 책 표준 ToolNode.
# 도구별 가드(image_path 필수, skin_scores 필수 등)는 각 도구 함수 본문에서 직접 수행.
act = ToolNode([skin_analyze_tool, recommend_treatment_db_tool, search_pubmed_tool])


def inject_recommend_call(state: BeautyAgentState) -> dict[str, Any]:
    """진단(skin_scores) 후 추천 의도일 때, LLM 판단과 무관하게 recommend_treatment_db를 1회 강제.

    gpt-4o-mini가 진단 다음에 도구를 건너뛰고 시술명을 임의로 지어내는(환각) 일을 막는다.
    이 강제 호출로 실제 DB 추천이 히스토리에 들어와야 think가 사실 기반으로 리포트를 쓴다.
    db_forced 플래그로 1회만 실행돼 무한 루프를 방지한다.
    """
    call_id = f"forced_recommend_{uuid.uuid4().hex[:8]}"
    tool_call = {"name": "recommend_treatment_db", "args": {}, "id": call_id}
    msg = AIMessage(
        content="진단 결과를 바탕으로 매칭되는 시술을 조회하겠습니다.",
        tool_calls=[tool_call],
    )
    return {
        "messages": [msg],
        "actions": [dict(tool_call)],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "db_forced": True,
    }


def inject_pubmed_call(state: BeautyAgentState) -> dict[str, Any]:
    """시술 추천(db_recommendations) 직후, LLM 판단과 무관하게 search_pubmed를 1회 강제.

    gpt-4o-mini가 추천 다음 단계로 근거 검색을 안정적으로 이어가지 못하므로,
    추천에 학술 근거를 항상 덧붙이도록 그래프 차원에서 결정적으로 tool_call을 발행한다.
    pubmed_forced 플래그로 1회만 실행돼 무한 루프를 방지한다.
    """
    call_id = f"forced_pubmed_{uuid.uuid4().hex[:8]}"
    tool_call = {"name": "search_pubmed", "args": {}, "id": call_id}
    msg = AIMessage(
        content="추천 시술의 학술적 근거를 확인하기 위해 PubMed 논문을 검색하겠습니다.",
        tool_calls=[tool_call],
    )
    return {
        "messages": [msg],
        "actions": [dict(tool_call)],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "pubmed_forced": True,
    }


def observe(state: BeautyAgentState) -> dict[str, Any]:
    """가장 최근 ToolMessage들을 observations 리스트에 적재."""
    messages = state.get("messages") or []
    collected: list[str] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            text = extract_text(msg.content)
            if text:
                collected.append(text)
        elif isinstance(msg, AIMessage):
            break
    if not collected:
        return {}
    collected.reverse()
    return {"observations": collected}


def finish(state: BeautyAgentState) -> dict[str, Any]:
    """최종 답변 content를 final_answer로 발췌하고 is_complete=True."""
    last_ai_text = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            last_ai_text = extract_text(getattr(msg, "content", "")) or last_ai_text
            if last_ai_text:
                break
    return {
        "final_answer": last_ai_text,
        "is_complete": True,
    }


def compress(state: BeautyAgentState) -> dict[str, Any]:
    """db/pubmed raw 결과를 LLM으로 사실 압축해 compressed_summary에 저장.

    route_from_start가 데이터 충분(has_db or has_pub)을 보장한 뒤에만 진입.
    """
    db_recs = state.get("db_recommendations") or []
    pubmed_recs = state.get("pubmed_recommendations") or []

    parts: list[str] = []
    if db_recs:
        parts.append(
            f"[DB 추천 시술 ({len(db_recs)}건)]\n" + "\n".join(
                f"- {r.get('region_ko')} ({r.get('score')}점) → code {r.get('code')}: "
                f"{r.get('treatment')}"
                + (f"\n  안내: {(r.get('customer_desc') or '').strip()}" if r.get('customer_desc') else "")
                for r in db_recs
            )
        )
    if pubmed_recs:
        parts.append(
            f"[PubMed 논문 근거 ({len(pubmed_recs)}건)]\n" + "\n".join(
                f"- {r.get('region_ko')} ({r.get('score')}점, {r.get('severity')}): "
                f"{r.get('title')} — {r.get('authors')} {r.get('year')} (PMID {r.get('pmid')}) "
                f"abstract: {r.get('abstract')}"
                for r in pubmed_recs
            )
        )

    llm = load_llm()
    response = llm.invoke([
        SystemMessage(content=COMPRESS_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ])
    return {"compressed_summary": extract_text(getattr(response, "content", ""))}


def final_report(state: BeautyAgentState) -> dict[str, Any]:
    """누적 state(skin_scores + compressed_summary 또는 raw db/pubmed)를 종합해
    최종 환자용 레포트를 작성하고 대화를 종료한다.

    도구는 호출하지 않으며 LLM이 텍스트 답변만 생성한다. is_complete=True.
    누적 자료가 비어 있으면 그 사실을 명시적으로 안내한다.
    """
    skin_scores = state.get("skin_scores") or {}
    db_recs = state.get("db_recommendations") or []
    pubmed_recs = state.get("pubmed_recommendations") or []
    summary = state.get("compressed_summary")
    history_text = latest_human_message_text(state.get("messages") or [])

    parts: list[str] = ["다음 누적 자료를 바탕으로 환자용 최종 레포트를 작성하세요."]
    if skin_scores.get("raw_scores"):
        aggregated = aggregate_regions(skin_scores["raw_scores"])
        score_text = ", ".join(
            f"{r['region_ko']} {r['score']:.1f}점" for r in aggregated
        ) or "(점수 없음)"
        worst = ", ".join(
            f"{r['region_ko']} {r['score']:.1f}점" for r in aggregated[:3]
        )
        parts.append(
            "\n[피부 진단 결과] (점수 0~100, 낮을수록 심각 / 아래 값을 그대로 사용하고 새 숫자를 만들지 마세요)\n"
            f"- 추정 연령: {skin_scores.get('age')}\n"
            f"- 성별: {skin_scores.get('gender_input') or state.get('gender')}\n"
            f"- 부위별 점수: {score_text}\n"
            f"- 가장 심각한(점수 낮은) 부위 top3: {worst}"
        )
    if summary:
        parts.append(f"\n[추천·근거 요약]\n{summary}")
    else:
        # compress 노드를 거치지 않은 경우의 fallback (raw 직접 포매팅)
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
    directive = severe_untreated_directive(state)
    if directive:
        parts.append(f"\n{directive}")
    if history_text:
        parts.append(f"\n[사용자의 직전 요청]\n{history_text}")

    llm = load_llm(  # bind_tools 안 함 — 순수 텍스트
        temperature=AI_TEMPERATURE_CHAT,
        frequency_penalty=0.3,
        max_tokens=2500,
    )
    response = llm.invoke([
        SystemMessage(content=FINAL_REPORT_PROMPT),
        HumanMessage(content="\n".join(parts)),
    ])

    return {
        "messages":     [response],
        "final_answer": extract_text(getattr(response, "content", "")),
        "is_complete":  True,
    }


def insufficient_response(state) -> dict:
    """레포트 요청인데 데이터가 부족할 때 후속 질문을 던지고 END.

    같은 thread_id로 다음 invoke가 들어오면 사용자 답변이 messages에 얹혀
    자연스럽게 think → ReAct 사이클로 이어진다.
    """
    completed: list[str] = []
    if state.get("skin_scores"):
        completed.append("진단")
    if state.get("db_recommendations"):
        completed.append("시술 추천")
    if state.get("pubmed_recommendations"):
        completed.append("논문 근거")

    progress_line = (
        f"지금까지 완료된 단계: {', '.join(completed)}."
        if completed
        else "아직 진행된 단계가 없습니다."
    )

    msg = AIMessage(content=(
        "최종 레포트를 작성하려면 진단 결과와 시술 추천/논문 근거가 필요합니다.\n"
        f"{progress_line}\n"
        "어떤 부분부터 진행하시겠어요? 사진을 업로드해 주시거나 진단부터 시작해도 됩니다."
    ))
    text = extract_text(msg.content)
    return {"messages": [msg], "final_answer": text, "is_complete": True}
