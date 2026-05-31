"""state.skin_scores를 받아 PubMed에서 시술 근거 논문을 RAG 검색하는 tool.

고도화된 RAG 설계:
- 입력: 없음 (InjectedState로 skin_scores + db_recommendations 읽음)
- 출력: state.pubmed_recommendations 갱신 + LLM에 노출할 자연어 요약
- Flow per 부위:
    1) raw 점수 → 부위 평균 + severity 라벨링
    2) (LLM) 환자 컨텍스트 + DB 추천 시술을 반영한 동적 PubMed 검색어 생성
       - 실패 시 부위별 고정 키워드(REGION_TO_PUBMED_QUERY)로 graceful fallback
    3) PubMed 쿼리 정교화: humans/최근 연도/연구유형(RCT·메타분석·체계적고찰) 필터
       - strict → broad → broadest 단계적 완화로 결과 확보
    4) NCBI E-utilities: esearch(후보 풀 확대) → efetch(XML)
    5) 배치 임베딩으로 query-abstract cosine 랭킹 → 임계값 통과 후보 top-k
    6) (LLM) 임상 관련성 리랭킹으로 최종 정렬
- 점수 가장 낮은(=가장 심각한) top-N 부위만 검색 (PubMed/임베딩 비용 trade-off)
- esearch/efetch/임베딩/LLM 쿼리는 디스크+메모리 캐시로 재호출 비용 절감
- 조회된 논문은 PMID 기준 영속 코퍼스(pubmed_corpus)에 중복 없이 누적되고,
  임베딩을 함께 보관해 이후 검색에서 로컬 cosine 후보로 재활용된다
- Hard guard: skin_scores가 없으면 skin_analyze를 먼저 호출하라는 에러

이 도구는 AuraDB(`recommend_treatment_db`)의 추천 시술을 검색어에 활용하지만,
최종 결과는 독립적인 pubmed_recommendations로 저장된다. 두 결과의 가중치 통합(debate)은
graph 단의 별도 orchestrator/노드 책임.
"""
from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Annotated, Any

import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from config import PROJECT_ROOT
from tools import pubmed_corpus
from tools.skin_analyze import aggregate_regions

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIMENSIONS = int(os.getenv("OPENAI_EMBED_DIMENSIONS", "1536"))

# ───────────────────────── 튜닝 가능한 설정(env) ─────────────────────────
_REGIONS_TO_QUERY = int(os.getenv("PUBMED_REGIONS", "3"))        # 검색할 부위 수 (top-N 심각)
_PMID_POOL_PER_REGION = int(os.getenv("PUBMED_PMID_POOL", "25")) # 부위당 esearch 후보 PMID 수
_TOP_K_PER_REGION = int(os.getenv("PUBMED_TOP_K", "2"))          # 부위당 최종 반환 논문 수
_MIN_SIMILARITY = float(os.getenv("PUBMED_MIN_SIMILARITY", "0.18"))  # cosine 최소 임계값
_RECENT_YEARS = int(os.getenv("PUBMED_RECENT_YEARS", "12"))      # 최근 N년 우선
_USE_LLM_QUERY = os.getenv("PUBMED_USE_LLM_QUERY", "1") != "0"
_USE_LLM_RERANK = os.getenv("PUBMED_USE_LLM_RERANK", "1") != "0"
_RERANK_POOL = int(os.getenv("PUBMED_RERANK_POOL", "6"))         # LLM 리랭킹에 넘길 후보 수
_CORPUS_LOCAL_K = int(os.getenv("PUBMED_CORPUS_LOCAL_K", "5"))   # 로컬 코퍼스에서 끌어올 후보 수
_REQ_TIMEOUT_SEC = 20

# 부위 region 키 → 영문 PubMed 검색어 (LLM 동적 생성 실패 시 fallback)
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

# OpenAI client는 lazy init (.env 로딩 순서 안전)
_openai_client = None


# ───────────────────────── 디스크+메모리 캐시 ─────────────────────────

_CACHE_FILE = PROJECT_ROOT / ".cache" / "pubmed_rag_cache.json"
_cache: dict | None = None
_cache_dirty = False


def _cache_load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:  # noqa: BLE001 - 캐시 없거나 손상 시 빈 캐시
            _cache = {}
        for ns in ("esearch", "efetch", "emb", "llmq"):
            _cache.setdefault(ns, {})
    return _cache


def _cache_get(ns: str, key: str):
    return _cache_load().get(ns, {}).get(key)


def _cache_set(ns: str, key: str, value) -> None:
    global _cache_dirty
    _cache_load().setdefault(ns, {})[key] = value
    _cache_dirty = True


def _cache_flush() -> None:
    global _cache_dirty
    if not _cache_dirty or _cache is None:
        return
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        _cache_dirty = False
    except Exception:  # noqa: BLE001 - 캐시 저장 실패는 치명적이지 않음
        pass


def _hkey(*parts) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ───────────────────────── 공통 헬퍼 ─────────────────────────

def _severity_keyword(score: float) -> str:
    if score <= 30:
        return "severe"
    if score <= 80:
        return "moderate"
    return "mild"


def _evidence_grade(study_type: str | None) -> str:
    """연구유형 → 근거등급(A/B/C). 메타분석·체계적고찰=A, RCT/임상시험=B, 그 외=C."""
    s = (study_type or "").lower()
    if "meta-analysis" in s or "systematic review" in s:
        return "A"
    if "randomized" in s or "clinical trial" in s:
        return "B"
    return "C"


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


def _parse_json_block(text: str) -> Any:
    """LLM 응답에서 JSON 블록(배열/객체)을 안전하게 파싱. 실패 시 None."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    # 첫 배열/객체 구간만 추출
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except Exception:  # noqa: BLE001
                continue
    return None


# ───────────────────────── 임베딩 (배치 + 캐시) ─────────────────────────

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 한 번의 API 호출로 임베딩. 텍스트별 캐시 적용.
    호출 실패 시 빈 벡터([])로 채워 호출자가 graceful 처리하도록 한다.
    """
    results: list[list[float]] = [[] for _ in texts]
    to_query: list[str] = []
    to_query_idx: list[int] = []

    for i, t in enumerate(texts):
        clean = (t or "").strip()[:8000] or " "  # 빈 문자열 임베딩 에러 방지
        key = _hkey(OPENAI_EMBED_MODEL, OPENAI_EMBED_DIMENSIONS, clean)
        cached = _cache_get("emb", key)
        if cached is not None:
            results[i] = cached
        else:
            to_query.append(clean)
            to_query_idx.append(i)

    if to_query:
        try:
            client = _get_openai_client()
            response = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=to_query,
                dimensions=OPENAI_EMBED_DIMENSIONS,
            )
            for j, item in enumerate(response.data):
                idx = to_query_idx[j]
                vec = item.embedding
                results[idx] = vec
                key = _hkey(OPENAI_EMBED_MODEL, OPENAI_EMBED_DIMENSIONS, to_query[j])
                _cache_set("emb", key, vec)
        except Exception:  # noqa: BLE001 - 임베딩 실패 시 빈 벡터 유지
            pass

    return results


# ───────────────────────── NCBI E-utilities (캐시) ─────────────────────────

def _esearch(query: str, max_results: int, recent: bool) -> list[str]:
    """NCBI esearch → relevance 정렬 PMID 리스트. recent=True면 최근 N년 필터."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if recent:
        this_year = datetime.now().year
        params["datetype"] = "pdat"
        params["mindate"] = str(this_year - _RECENT_YEARS)
        params["maxdate"] = str(this_year)
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    cache_key = _hkey("esearch", query, max_results, recent)
    cached = _cache_get("esearch", cache_key)
    if cached is not None:
        return cached

    url = f"{PUBMED_BASE}/esearch.fcgi"
    response = requests.get(url, params=params, timeout=_REQ_TIMEOUT_SEC)
    response.raise_for_status()
    idlist = response.json().get("esearchresult", {}).get("idlist", [])
    _cache_set("esearch", cache_key, idlist)
    return idlist


def _efetch(pmids: list[str]) -> list[dict]:
    """PMID들 → abstract metadata 리스트. PMID 집합 단위로 캐시."""
    if not pmids:
        return []

    cache_key = _hkey("efetch", ",".join(sorted(pmids)))
    cached = _cache_get("efetch", cache_key)
    if cached is not None:
        pubmed_corpus.upsert(cached)  # 캐시 히트여도 코퍼스 누적 보장(dedup)
        return cached

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
    parsed = _parse_efetch_xml(response.text)
    _cache_set("efetch", cache_key, parsed)
    # PMID 기준 영속 코퍼스에 누적(중복 제거). 캐시와 달리 논문 단위로 dedup된다.
    pubmed_corpus.upsert(parsed)
    return parsed


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

        # 근거수준 높은 연구유형을 우선 노출 (메타분석/체계적고찰/RCT)
        pub_types = [el.text for el in pub_type_els if el.text]
        study_type = None
        for preferred in ("Meta-Analysis", "Systematic Review", "Randomized Controlled Trial", "Clinical Trial"):
            if preferred in pub_types:
                study_type = preferred
                break
        if study_type is None and pub_types:
            study_type = pub_types[0]

        parsed.append({
            "pmid":       pmid_el.text if pmid_el is not None else None,
            "title":      title_el.text if title_el is not None else None,
            "abstract":   abstract,
            "year":       year,
            "authors":    ", ".join(author_names) if author_names else None,
            "study_type": study_type,
        })
    return parsed


# ───────────────────────── 동적 쿼리 생성 (LLM) ─────────────────────────

_QUERY_GEN_SYSTEM = (
    "You are a biomedical literature search assistant for aesthetic dermatology. "
    "For each skin concern, craft optimized English PubMed search terms. "
    "Return ONLY a JSON array. Each element must be an object with keys: "
    '"region" (echo the given region key exactly), '
    '"core" (concise free-text keywords describing the concern and its aesthetic treatment), '
    '"treatment_terms" (device/drug/procedure names for the recommended treatment, or empty string), '
    '"mesh" (array of up to 3 relevant MeSH headings). '
    "Keep terms specific and clinical. No prose, JSON only."
)


def _llm_generate_queries(targets: list[dict], patient_ctx: str) -> dict[str, dict]:
    """부위 목록 + 환자 컨텍스트 → 부위별 검색어 사양(dict). 실패 시 빈 dict."""
    if not _USE_LLM_QUERY or not targets:
        return {}

    payload_lines = [f"Patient context: {patient_ctx}", "Concerns:"]
    for t in targets:
        treat = t.get("treatment") or "(no specific treatment provided)"
        payload_lines.append(
            f"- region={t['region']} | concern={t['region_ko']} | "
            f"severity={t['severity']} | recommended_treatment={treat}"
        )
    user_text = "\n".join(payload_lines)

    cache_key = _hkey("llmq", _QUERY_GEN_SYSTEM, user_text)
    cached = _cache_get("llmq", cache_key)
    if cached is not None:
        return cached

    try:
        from agent.helpers import extract_text, load_llm  # 지연 import (순환 방지)
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = load_llm()
        response = llm.invoke([
            SystemMessage(content=_QUERY_GEN_SYSTEM),
            HumanMessage(content=user_text),
        ])
        parsed = _parse_json_block(extract_text(getattr(response, "content", "")))
        if not isinstance(parsed, list):
            return {}
        specs: dict[str, dict] = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("region"):
                specs[item["region"]] = {
                    "core": (item.get("core") or "").strip(),
                    "treatment_terms": (item.get("treatment_terms") or "").strip(),
                    "mesh": [m for m in (item.get("mesh") or []) if isinstance(m, str)][:3],
                }
        _cache_set("llmq", cache_key, specs)
        return specs
    except Exception:  # noqa: BLE001 - 쿼리 생성 실패 시 fallback 사용
        return {}


def _build_pubmed_query(spec: dict, level: str) -> str:
    """검색어 사양 + 완화 단계(level) → PubMed 쿼리 문자열.
    level: 'strict'(연구유형+humans) → 'broad'(humans) → 'broadest'(코어만).
    """
    core = spec.get("core") or ""
    treatment = spec.get("treatment_terms") or ""
    mesh = spec.get("mesh") or []

    base = f"({core})" if core else ""
    if treatment:
        base = f"(({core}) OR ({treatment}))" if core else f"({treatment})"
    if mesh and level != "broadest":
        mesh_clause = " OR ".join(f'"{m}"[MeSH Terms]' for m in mesh)
        base = f"({base} AND ({mesh_clause}))" if base else f"({mesh_clause})"

    if level == "broadest":
        return base or core

    parts = [base, "Humans[MeSH Terms]"]
    if level == "strict":
        parts.append(
            "(Randomized Controlled Trial[Publication Type] OR Meta-Analysis[Publication Type] "
            "OR Systematic Review[Publication Type] OR Clinical Trial[Publication Type])"
        )
    return " AND ".join(p for p in parts if p)


def _merge_corpus_candidates(query: str, articles: list[dict]) -> list[dict]:
    """누적 코퍼스에서 query와 유사한 논문을 끌어와 PubMed 결과와 병합(PMID dedup).

    이미 확보한 PMID는 PubMed 쪽(메타데이터가 더 완전한) 버전을 유지하고,
    코퍼스에만 있는 신규 후보만 뒤에 추가한다. 임베딩이 없으면(=콜드 스타트)
    코퍼스 검색은 빈 결과를 반환하므로 기존 동작과 동일하다.
    """
    if not (pubmed_corpus.CORPUS_ENABLED and query and _CORPUS_LOCAL_K > 0):
        return articles

    q_vec = _embed_batch([query])[0]  # 동일 텍스트는 emb 캐시로 재사용됨
    if not q_vec:
        return articles

    local_hits = pubmed_corpus.search(q_vec, top_k=_CORPUS_LOCAL_K, min_sim=_MIN_SIMILARITY)
    if not local_hits:
        return articles

    seen = {str(a.get("pmid")) for a in articles if a.get("pmid")}
    merged = list(articles)
    for hit in local_hits:
        pmid = str(hit.get("pmid")) if hit.get("pmid") else None
        if pmid and pmid not in seen:
            merged.append(hit)
            seen.add(pmid)
    return merged


def _retrieve_for_region(spec: dict) -> tuple[str, list[dict], str]:
    """strict→broad→broadest 순으로 후보를 확보. (사용쿼리, articles, level) 반환."""
    last_query = ""
    for level in ("strict", "broad", "broadest"):
        query = _build_pubmed_query(spec, level)
        if not query:
            continue
        last_query = query
        recent = level != "broadest"
        try:
            pmids = _esearch(query, _PMID_POOL_PER_REGION, recent=recent)
        except (requests.RequestException, ValueError):
            continue
        if not pmids:
            continue
        try:
            articles = _efetch(pmids)
        except (requests.RequestException, ET.ParseError):
            continue
        if articles:
            return query, articles, level
    return last_query, [], "none"


# ───────────────────────── 랭킹 (임베딩 + LLM 리랭킹) ─────────────────────────

def _rank_by_embedding(query: str, articles: list[dict]) -> list[dict]:
    """query와 각 article(title+abstract)을 배치 임베딩해 cosine 유사도로 정렬."""
    docs = [
        ((a.get("title") or "") + ". " + (a.get("abstract") or "")).strip()
        for a in articles
    ]
    vectors = _embed_batch([query] + docs)
    q_vec = vectors[0]
    doc_vecs = vectors[1:]

    if not q_vec:  # 임베딩 실패 → esearch relevance 순서 유지
        for a in articles:
            a["similarity"] = None
        return articles

    for a, dv in zip(articles, doc_vecs):
        a["similarity"] = _cosine(q_vec, dv) if dv else 0.0
        # 계산한 문서 임베딩을 코퍼스에 저장 → 이후 로컬 벡터 검색에 재사용
        if dv:
            pubmed_corpus.set_embedding(a.get("pmid"), dv)
    articles.sort(key=lambda x: x.get("similarity") or 0.0, reverse=True)
    return articles


_RERANK_SYSTEM = (
    "You are a clinical evidence reviewer. Given a patient's skin concern and candidate "
    "PubMed articles, score each candidate's relevance as treatment evidence from 0.0 to 1.0. "
    "Prioritize human studies, higher evidence levels, and direct relevance to the concern and "
    "recommended treatment. Return ONLY a JSON array of objects {\"index\": int, \"score\": float}, "
    "sorted by score descending."
)


def _llm_rerank(context: str, articles: list[dict]) -> list[dict]:
    """cosine 상위 후보를 LLM 임상 관련성으로 재정렬. 실패 시 입력 순서 유지."""
    if not _USE_LLM_RERANK or len(articles) <= 1:
        return articles

    pool = articles[:_RERANK_POOL]
    lines = [f"Concern context: {context}", "Candidates:"]
    for i, a in enumerate(pool):
        snippet = (a.get("abstract") or "").replace("\n", " ")[:300]
        lines.append(
            f"[{i}] type={a.get('study_type')} year={a.get('year')} "
            f"title={a.get('title')} :: {snippet}"
        )
    user_text = "\n".join(lines)

    try:
        from agent.helpers import extract_text, load_llm  # 지연 import
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = load_llm()
        response = llm.invoke([
            SystemMessage(content=_RERANK_SYSTEM),
            HumanMessage(content=user_text),
        ])
        parsed = _parse_json_block(extract_text(getattr(response, "content", "")))
        if not isinstance(parsed, list):
            return articles

        reranked: list[dict] = []
        seen: set[int] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                pool[idx]["rerank_score"] = item.get("score")
                reranked.append(pool[idx])
                seen.add(idx)
        # LLM이 누락한 후보 + pool 밖 후보는 뒤에 보존
        for i, a in enumerate(pool):
            if i not in seen:
                reranked.append(a)
        reranked.extend(articles[_RERANK_POOL:])
        return reranked
    except Exception:  # noqa: BLE001 - 리랭킹 실패 시 cosine 순서 유지
        return articles


# ───────────────────────── 출력 포맷 ─────────────────────────

def _format_pubmed_summary(recs: list[dict], errors: list[dict]) -> str:
    if not recs:
        body = "PubMed 검색 결과를 얻지 못했습니다."
    else:
        lines = [f"PubMed RAG 검색 결과 (부위별 top {_TOP_K_PER_REGION} 논문, 근거등급 A>B>C)"]
        for r in recs:
            header_extra = []
            if r.get("treatment"):
                header_extra.append(f"추천시술: {r['treatment']}")
            head_extra_str = f" / {' / '.join(header_extra)}" if header_extra else ""
            lines.append("")
            lines.append(f"- {r['region_ko']} ({r['score']:.1f}점, {r['severity']}){head_extra_str}")

            for art in r.get("articles", []):
                citation_parts = []
                if art.get("authors"):
                    citation_parts.append(art["authors"])
                if art.get("year"):
                    citation_parts.append(str(art["year"]))
                if art.get("study_type"):
                    citation_parts.append(art["study_type"])
                citation = " · ".join(citation_parts) if citation_parts else "출처 정보 없음"
                pmid = art.get("pmid") or "?"
                title = (art.get("title") or "(제목 없음)").strip()
                grade = art.get("evidence_grade") or "-"
                abstract_snippet = (art.get("abstract") or "").strip().replace("\n", " ")
                if len(abstract_snippet) > 300:
                    abstract_snippet = abstract_snippet[:300] + "…"

                lines.append(f"  · [{grade}] {title}")
                lines.append(f"    출처: {citation} | PMID {pmid}")
                if abstract_snippet:
                    lines.append(f"    초록 요약: {abstract_snippet}")
        body = "\n".join(lines)

    if errors:
        err_str = "; ".join(f"{e.get('region')}: {e.get('reason')}" for e in errors)
        body += f"\n\n[조회 오류] {err_str}"
    return body


def _build_patient_context(state: dict, skin_scores: dict) -> str:
    age = skin_scores.get("age")
    gender = skin_scores.get("gender_input") or state.get("gender")
    parts = []
    if gender:
        parts.append(f"gender={gender}")
    if age is not None:
        parts.append(f"estimated_skin_age={int(round(age))}")
    return ", ".join(parts) if parts else "no demographic info"


def _treatment_map(state: dict) -> dict[str, str]:
    """db_recommendations에서 region → 추천 시술명 매핑. 관리 불필요(_0) 코드는 제외."""
    mapping: dict[str, str] = {}
    for rec in state.get("db_recommendations") or []:
        region = rec.get("region")
        code = rec.get("code") or ""
        treatment = (rec.get("treatment") or "").strip()
        if region and treatment and not code.endswith("_0"):
            mapping[region] = treatment
    return mapping


def _target_regions_from_db(state: dict) -> list[dict]:
    """DB 추천 결과가 있으면 실제 시술이 필요한(_0이 아닌) 부위만 PubMed 검색 대상으로 삼는다.

    추천 리포트의 Step 3 근거는 Step 2에서 실제로 권한 시술에만 붙어야 한다.
    진단 점수가 낮더라도 DB가 '시술 불필요(_0)'로 매칭한 부위는 검색 대상에서 제외한다.
    """
    targets: list[dict] = []
    for rec in state.get("db_recommendations") or []:
        code = rec.get("code") or ""
        if code.endswith("_0"):
            continue
        treatment = (rec.get("treatment") or "").strip()
        if not treatment:
            continue
        score = rec.get("score")
        if score is None:
            continue
        targets.append({
            "region":    rec.get("region"),
            "region_ko": rec.get("region_ko"),
            "score":     score,
            "severity":  _severity_keyword(score),
            "treatment": treatment,
        })
    return sorted(
        [t for t in targets if t.get("region") and t.get("region_ko") and t.get("score") is not None],
        key=lambda t: t["score"],
    )


# ───────────────────────── tool ─────────────────────────

@tool
def search_pubmed(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """state.skin_scores 기반으로 PubMed에서 시술 근거 논문을 RAG 검색합니다.
    db_recommendations가 있으면 실제 시술이 필요한(_0이 아닌) 추천 부위만 검색합니다.
    db_recommendations가 없을 때만 점수가 가장 낮은(=가장 심각한) 상위 부위를 fallback으로 검색합니다.
    환자 컨텍스트와 DB 추천 시술을 반영한 동적 검색어를 생성하고(LLM), humans/최근성/연구유형 필터로 PubMed를 검색합니다.
    배치 임베딩 cosine + LLM 임상 리랭킹으로 부위별 top-k 논문을 추려냅니다.

    추가 인자는 받지 않으며 state.skin_scores.raw_scores와 (있다면) db_recommendations를 활용합니다.
    반드시 skin_analyze를 먼저 호출한 뒤 사용하세요. recommend_treatment_db를 먼저 호출하면
    추천 시술을 검색어에 반영해 근거 매칭 품질이 올라갑니다.

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

    treatments = _treatment_map(state)
    patient_ctx = _build_patient_context(state, skin_scores)

    # 검색 대상 부위 메타데이터 구성
    targets = _target_regions_from_db(state)
    if not targets and not state.get("db_recommendations"):
        # DB 추천 없이 직접 근거를 요청한 경우만 기존처럼 점수 하위 부위를 fallback 검색한다.
        target_regions = aggregate_regions(raw_scores)[:_REGIONS_TO_QUERY]
        for concern in target_regions:
            targets.append({
                "region":    concern["region"],
                "region_ko": concern["region_ko"],
                "score":     concern["score"],
                "severity":  _severity_keyword(concern["score"]),
                "treatment": treatments.get(concern["region"], ""),
            })
    elif not targets:
        return Command(update={"messages": [_make_tool_message(
            "DB 추천 결과에서 실제 시술이 필요한 부위가 없어 PubMed 근거 검색을 생략합니다.",
            tool_call_id,
        )]})

    # 1) 동적 쿼리 생성 (LLM, 1회 배치 호출)
    llm_specs = _llm_generate_queries(targets, patient_ctx)

    pubmed_recommendations: list[dict] = []
    errors: list[dict] = []

    for t in targets:
        region = t["region"]
        spec = llm_specs.get(region)
        if not spec or not (spec.get("core") or spec.get("treatment_terms")):
            # fallback: 고정 키워드 + severity
            fallback_core = REGION_TO_PUBMED_QUERY.get(region)
            if not fallback_core:
                errors.append({"region": region, "reason": "검색어 생성 실패"})
                continue
            spec = {
                "core": f"{fallback_core} {t['severity']}",
                "treatment_terms": t["treatment"],
                "mesh": [],
            }

        # 2~3) 단계적 완화 검색 + 후보 확보
        used_query, articles, level = _retrieve_for_region(spec)

        # 3.5) 누적 코퍼스에서 유사 논문 병합 (PubMed가 비어도 로컬 후보로 보강)
        merge_query = used_query or _build_pubmed_query(spec, "broadest")
        articles = _merge_corpus_candidates(merge_query, articles)
        if not used_query and articles:
            used_query, level = merge_query, "corpus"

        if not articles:
            errors.append({"region": region, "reason": "PubMed/코퍼스 결과 없음"})
            continue

        # 4) 배치 임베딩 cosine 랭킹
        ranked = _rank_by_embedding(used_query, articles)

        # 임계값 필터 (임베딩 성공 시에만 적용)
        if ranked and ranked[0].get("similarity") is not None:
            filtered = [a for a in ranked if (a.get("similarity") or 0.0) >= _MIN_SIMILARITY]
            ranked = filtered or ranked[:1]  # 전부 미달이면 최상위 1개는 보존

        # 5) LLM 임상 리랭킹
        context = (
            f"{t['region_ko']} ({t['severity']}); "
            f"recommended treatment: {t['treatment'] or 'N/A'}; {patient_ctx}"
        )
        ranked = _llm_rerank(context, ranked)

        top_articles = ranked[:_TOP_K_PER_REGION]
        for art in top_articles:
            art["evidence_grade"] = _evidence_grade(art.get("study_type"))
            if isinstance(art.get("abstract"), str):
                art["abstract"] = art["abstract"][:600]

        best = top_articles[0]
        # 하위호환: compress/final_report가 참조하는 평면 필드는 best 기준으로 유지
        pubmed_recommendations.append({
            "region":     region,
            "region_ko":  t["region_ko"],
            "score":      t["score"],
            "severity":   t["severity"],
            "treatment":  t["treatment"],
            "query_used": used_query,
            "query_level": level,
            "pmid":       best.get("pmid"),
            "title":      best.get("title"),
            "authors":    best.get("authors"),
            "year":       best.get("year"),
            "study_type": best.get("study_type"),
            "evidence_grade": best.get("evidence_grade"),
            "abstract":   best.get("abstract"),
            "similarity": best.get("similarity"),
            "articles":   top_articles,  # 부위별 top-k 전체
        })

    _cache_flush()
    pubmed_corpus.flush()

    summary = _format_pubmed_summary(pubmed_recommendations, errors)
    if pubmed_corpus.CORPUS_ENABLED:
        cstats = pubmed_corpus.stats()
        summary += (
            f"\n\n[논문 코퍼스] 누적 {cstats['count']}편"
            f" (임베딩 보유 {cstats['with_embedding']}편) — 검색할수록 중복 없이 누적됩니다."
        )
    return Command(update={
        "pubmed_recommendations": pubmed_recommendations,
        "messages":               [_make_tool_message(summary, tool_call_id)],
    })
