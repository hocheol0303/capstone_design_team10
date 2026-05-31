"""PubMed 논문 영속 코퍼스 (PMID 키 기반, 중복 제거 + 벡터 검색).

기존 `.cache/pubmed_rag_cache.json`은 "동일 입력 재호출 비용 절감"용 key-value 캐시라
같은 논문이 여러 efetch 항목에 중복 저장되고 검색도 불가능하다.

이 모듈은 그와 분리된 **누적 지식 베이스**를 제공한다:
- PMID를 1차 키로 사용 → 같은 논문은 한 번만 저장(중복 제거)
- 논문별 임베딩 벡터를 함께 보관 → 새 쿼리에 대해 로컬 cosine 검색 가능
- upsert 시 메타데이터를 보강(기존 비어있던 필드만 채움)하고 접근 통계를 누적

저장 위치: `.cache/pubmed_corpus.json`
구조:
{
  "papers": {
    "<pmid>": {
      "pmid", "title", "abstract", "year", "authors", "study_type",
      "embedding": [float, ...] | None,
      "first_seen": iso8601, "last_seen": iso8601, "seen_count": int,
      "queries": [str, ...]   # 이 논문을 표면화한 최근 쿼리(상한 있음)
    }, ...
  }
}
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from config import PROJECT_ROOT

_CORPUS_FILE = PROJECT_ROOT / ".cache" / "pubmed_corpus.json"

# 코퍼스 사용 여부 / 임베딩 저장 여부 (env로 토글)
CORPUS_ENABLED = os.getenv("PUBMED_CORPUS_ENABLED", "1") != "0"
_STORE_EMBEDDING = os.getenv("PUBMED_CORPUS_STORE_EMB", "1") != "0"
# 논문당 보관할 최근 쿼리 수 상한 (파일 비대화 방지)
_MAX_QUERIES_PER_PAPER = int(os.getenv("PUBMED_CORPUS_MAX_QUERIES", "8"))

_corpus: dict | None = None
_dirty = False

# 메타데이터 보강 대상 필드 (값이 있으면 보존, 없을 때만 채움)
_META_FIELDS = ("title", "abstract", "year", "authors", "study_type")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> dict:
    global _corpus
    if _corpus is None:
        try:
            with open(_CORPUS_FILE, encoding="utf-8") as f:
                _corpus = json.load(f)
        except Exception:  # noqa: BLE001 - 파일 없거나 손상 시 빈 코퍼스
            _corpus = {}
        _corpus.setdefault("papers", {})
    return _corpus


def upsert(articles: list[dict], query: str | None = None) -> int:
    """논문 리스트를 PMID 기준으로 코퍼스에 누적(중복 제거).

    Returns:
        새로 추가된 논문 수.
    """
    if not CORPUS_ENABLED or not articles:
        return 0
    global _dirty
    papers = _load()["papers"]
    now = _now()
    added = 0

    for art in articles:
        pmid = art.get("pmid")
        if not pmid:
            continue
        pmid = str(pmid)
        entry = papers.get(pmid)
        if entry is None:
            entry = {
                "pmid": pmid,
                "embedding": None,
                "first_seen": now,
                "seen_count": 0,
                "queries": [],
            }
            papers[pmid] = entry
            added += 1

        # 메타데이터 보강: 기존에 비어있던 값만 채움
        for field in _META_FIELDS:
            val = art.get(field)
            if val not in (None, "") and entry.get(field) in (None, ""):
                entry[field] = val

        entry["last_seen"] = now
        entry["seen_count"] = int(entry.get("seen_count", 0)) + 1

        if query:
            q_list = entry.setdefault("queries", [])
            if query not in q_list:
                q_list.append(query)
                if len(q_list) > _MAX_QUERIES_PER_PAPER:
                    del q_list[: len(q_list) - _MAX_QUERIES_PER_PAPER]

        _dirty = True

    return added


def set_embedding(pmid: str | None, vector: list[float] | None) -> None:
    """논문 임베딩 저장(로컬 벡터 검색용). PMID가 코퍼스에 없으면 무시."""
    if not (CORPUS_ENABLED and _STORE_EMBEDDING) or not pmid or not vector:
        return
    global _dirty
    entry = _load()["papers"].get(str(pmid))
    if entry is not None and entry.get("embedding") is None:
        entry["embedding"] = vector
        _dirty = True


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(query_vec: list[float], top_k: int, min_sim: float = 0.0) -> list[dict]:
    """query 임베딩으로 코퍼스를 cosine 검색. 임베딩 없는 논문은 제외.

    Returns:
        유사도 내림차순 top_k 논문(dict 복사본, embedding 제외, similarity/source 포함).
    """
    if not CORPUS_ENABLED or not query_vec or top_k <= 0:
        return []

    scored: list[tuple[float, dict]] = []
    for entry in _load()["papers"].values():
        vec = entry.get("embedding")
        if not vec:
            continue
        sim = _cosine(query_vec, vec)
        if sim < min_sim:
            continue
        scored.append((sim, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[dict] = []
    for sim, entry in scored[:top_k]:
        results.append({
            "pmid":       entry.get("pmid"),
            "title":      entry.get("title"),
            "abstract":   entry.get("abstract"),
            "year":       entry.get("year"),
            "authors":    entry.get("authors"),
            "study_type": entry.get("study_type"),
            "similarity": sim,
            "source":     "corpus",
        })
    return results


def stats() -> dict:
    papers = _load()["papers"]
    with_emb = sum(1 for e in papers.values() if e.get("embedding"))
    return {"count": len(papers), "with_embedding": with_emb}


def flush() -> None:
    global _dirty
    if not _dirty or _corpus is None:
        return
    try:
        _CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(_corpus, f, ensure_ascii=False)
        _dirty = False
    except Exception:  # noqa: BLE001 - 코퍼스 저장 실패는 치명적이지 않음
        pass
