"""state.skin_scores를 받아 PubMed에서 시술 근거 논문을 RAG 검색하는 tool.

Phase 4 설계:
- 입력: 없음 (InjectedState로 skin_scores 읽음)
- 출력: state.pubmed_recommendations 갱신 + LLM에 노출할 자연어 요약
- Flow per 부위:
    1) raw 점수 → 부위 평균 + severity 라벨링
    2) region 키 → 영문 PubMed 검색어 생성 (severity 포함)
    3) NCBI E-utilities: esearch (PMID 후보 N개, relevance 정렬) → efetch (XML)
    4) 각 abstract를 OpenAI 임베딩으로 벡터화, query와 cosine 유사도 랭킹 → top-1
- 점수 가장 낮은(=가장 심각한) **top 3** 부위만 검색 (PubMed/임베딩 비용 trade-off)
- Hard guard: skin_scores가 없으면 skin_analyze를 먼저 호출하라는 에러

이 도구는 AuraDB(`recommend_treatment_db`)와 **독립적**으로 동작한다.
두 결과의 가중치 기반 통합(debate)은 graph 단의 별도 orchestrator/노드 책임.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Annotated, Optional

import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from tools.skin_analyze import aggregate_regions

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIMENSIONS = int(os.getenv("OPENAI_EMBED_DIMENSIONS", "1536"))

# 부위 region 키 → 영문 PubMed 검색어 (시술 명사 포함, severity는 호출 시 추가)
REGION_TO_PUBMED_QUERY: dict[str, str] = {
    "pigmentation":        "facial pigmentation treatment laser",
    "forehead_wrinkle":    "forehead glabellar wrinkle botulinum toxin treatment",
    "eye_wrinkle":         "periorbital crow's feet wrinkle treatment",
    "nasolabial_fold":     "nasolabial fold filler treatment",
    "perioral_wrinkle":    "perioral wrinkles treatment",
    "volume_wrinkle":      "midface volume loss dermal filler",
    "homogenity_radiance": "skin radiance dullness treatment",
    "homogenity_texture":  "skin texture roughness treatment",
    "cheek_sagging":       "cheek sagging facial lifting HIFU",
    "chin_sagging":        "jowl jawline lifting treatment",
}

_REQ_TIMEOUT_SEC = 20
_PMID_POOL_PER_REGION = 5      # 부위당 esearch에서 가져올 PMID 수
_REGIONS_TO_QUERY = 3          # 임베딩 RAG로 추리는 부위 수 (top-N 심각)

# OpenAI client는 모듈 로드 시점에는 생성하지 않고 lazy init (.env 로딩 순서 안전)
_openai_client = None


# ───────────────────────── 헬퍼 ─────────────────────────

def _severity_keyword(score: float) -> str:
    if score <= 30:
        return "severe"
    if score <= 80:
        return "moderate"
    return "mild"


def _make_tool_message(content, tool_call_id: str) -> ToolMessage:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _cosine(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # 지연 import
        _openai_client = OpenAI()
    return _openai_client


def _embed(text: str) -> list[float]:
    """OpenAI text-embedding-3-small 호출. 호출 실패는 호출자에서 잡는다."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text,
        dimensions=OPENAI_EMBED_DIMENSIONS,
    )
    return response.data[0].embedding


def _esearch(query: str, max_results: int = 5) -> list[str]:
    """NCBI esearch → relevance 정렬 PMID 리스트."""
    url = f"{PUBMED_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    response = requests.get(url, params=params, timeout=_REQ_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> list[dict]:
    """PMID들 → abstract metadata 리스트."""
    if not pmids:
        return []
    url = f"{PUBMED_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    response = requests.get(url, params=params, timeout=_REQ_TIMEOUT_SEC)
    response.raise_for_status()
    return _parse_efetch_xml(response.text)


def _parse_efetch_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    parsed: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//Abstract/AbstractText")
        year_el = article.find(".//PubDate/Year") or article.find(".//PubDate/MedlineDate")
        pub_type_els = article.findall(".//PublicationType")
        author_els = article.findall(".//AuthorList/Author")

        author_names: list[str] = []
        for a in author_els[:3]:
            last = a.findtext("LastName") or ""
            init = a.findtext("Initials") or ""
            name = f"{last} {init}".strip()
            if name:
                author_names.append(name)
        if len(author_els) > 3 and author_names:
            author_names.append("et al.")

        year_text = year_el.text if year_el is not None and year_el.text else ""
        year: int | None = None
        if year_text:
            digits = "".join(ch for ch in year_text if ch.isdigit())
            if digits[:4].isdigit():
                year = int(digits[:4])

        abstract = " ".join((el.text or "") for el in abstract_parts).strip()
        study_type = pub_type_els[0].text if pub_type_els else None

        parsed.append({
            "pmid":       pmid_el.text if pmid_el is not None else None,
            "title":      title_el.text if title_el is not None else None,
            "abstract":   abstract,
            "year":       year,
            "authors":    ", ".join(author_names) if author_names else None,
            "study_type": study_type,
        })
    return parsed


def _rank_by_embedding(query: str, articles: list[dict]) -> list[dict]:
    """query와 각 article(title+abstract) 임베딩 cosine 유사도로 정렬.
    임베딩 호출 실패 시 esearch relevance 순서 유지.
    """
    try:
        q_vec = _embed(query)
    except Exception:  # noqa: BLE001
        return articles

    for article in articles:
        doc = ((article.get("title") or "") + ". " + (article.get("abstract") or "")).strip()
        if not doc:
            article["similarity"] = 0.0
            continue
        try:
            doc_vec = _embed(doc[:8000])  # text-embedding-3-small 토큰 한도 보호
            article["similarity"] = _cosine(q_vec, doc_vec)
        except Exception:  # noqa: BLE001
            article["similarity"] = 0.0
    articles.sort(key=lambda x: x.get("similarity") or 0.0, reverse=True)
    return articles


def _format_pubmed_summary(recs: list[dict], errors: list[dict]) -> str:
    if not recs:
        body = "PubMed 검색 결과를 얻지 못했습니다."
    else:
        lines = ["PubMed RAG 검색 결과 (부위별 top 1 논문)"]
        for r in recs:
            citation_parts = []
            if r.get("authors"):
                citation_parts.append(r["authors"])
            if r.get("year"):
                citation_parts.append(str(r["year"]))
            if r.get("study_type"):
                citation_parts.append(r["study_type"])
            citation = " · ".join(citation_parts) if citation_parts else "출처 정보 없음"
            pmid = r.get("pmid") or "?"
            title = (r.get("title") or "(제목 없음)").strip()
            abstract_snippet = (r.get("abstract") or "").strip().replace("\n", " ")
            if len(abstract_snippet) > 350:
                abstract_snippet = abstract_snippet[:350] + "…"

            lines.append("")
            lines.append(f"- {r['region_ko']} ({r['score']:.1f}점, {r['severity']})")
            lines.append(f"  · 논문: {title}")
            lines.append(f"  · 출처: {citation} | PMID {pmid}")
            if abstract_snippet:
                lines.append(f"  · 초록 요약: {abstract_snippet}")
        body = "\n".join(lines)

    if errors:
        err_str = "; ".join(f"{e.get('region')}: {e.get('reason')}" for e in errors)
        body += f"\n\n[조회 오류] {err_str}"
    return body


# ───────────────────────── tool ─────────────────────────

@tool
def search_pubmed(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """state.skin_scores 기반으로 PubMed에서 시술 근거 논문을 RAG 검색합니다.
    점수가 가장 낮은(=가장 심각한) 상위 3개 부위에 대해 영문 검색어를 만들고,
    각 부위마다 esearch → efetch → 임베딩 cosine 랭킹 top-1을 추려냅니다.

    추가 인자는 받지 않으며 state.skin_scores.raw_scores를 활용합니다.
    반드시 skin_analyze를 먼저 호출한 뒤 사용하세요.

    Returns:
        Command(update={...}) — state.pubmed_recommendations 갱신 + ToolMessage 삽입.
    """
    skin_scores = state.get("skin_scores") or {}
    raw_scores = skin_scores.get("raw_scores")
    if not raw_scores:
        return Command(update={"messages": [_make_tool_message(
            "진단 데이터(raw_scores)가 없습니다. 먼저 skin_analyze를 호출하세요.",
            tool_call_id,
        )]})

    target_regions = aggregate_regions(raw_scores)[:_REGIONS_TO_QUERY]

    pubmed_recommendations: list[dict] = []
    errors: list[dict] = []

    for concern in target_regions:
        region = concern["region"]
        region_ko = concern["region_ko"]
        score = concern["score"]
        en_keyword = REGION_TO_PUBMED_QUERY.get(region)
        if not en_keyword:
            errors.append({"region": region, "reason": "PubMed 키워드 매핑 없음"})
            continue

        severity = _severity_keyword(score)
        query = f"{en_keyword} {severity}"

        try:
            pmids = _esearch(query, max_results=_PMID_POOL_PER_REGION)
            articles = _efetch(pmids) if pmids else []
        except requests.RequestException as exc:
            errors.append({"region": region, "reason": f"PubMed API 실패: {exc}"})
            continue
        except ET.ParseError as exc:
            errors.append({"region": region, "reason": f"XML 파싱 실패: {exc}"})
            continue

        if not articles:
            errors.append({"region": region, "reason": "PubMed 결과 없음"})
            continue

        ranked = _rank_by_embedding(query, articles)
        best = ranked[0]

        pubmed_recommendations.append({
            "region":     region,
            "region_ko":  region_ko,
            "score":      score,
            "severity":   severity,
            "query_used": query,
            "pmid":       best.get("pmid"),
            "title":      best.get("title"),
            "authors":    best.get("authors"),
            "year":       best.get("year"),
            "study_type": best.get("study_type"),
            "abstract":   (best.get("abstract") or "")[:600],
            "similarity": best.get("similarity"),
        })

    summary = _format_pubmed_summary(pubmed_recommendations, errors)
    return Command(update={
        "pubmed_recommendations": pubmed_recommendations,
        "messages":               [_make_tool_message(summary, tool_call_id)],
    })
