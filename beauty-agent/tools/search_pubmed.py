import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool

from config import PUBMED_API_KEY, PUBMED_BASE_URL, USE_MOCK_PUBMED

_MOCK_PUBMED_RESULTS = {
    "forehead wrinkle severe": [
        {
            "pmid": "37291847",
            "title": "Botulinum Toxin for Forehead Wrinkle Treatment",
            "authors": "Kim SY et al.",
            "year": 2023,
            "study_type": "RCT",
            "sample_size": 98,
            "conclusion": "보톡스 4주 후 이마 주름 유의미한 개선",
        }
    ],
    "nasolabial fold severe filler": [
        {
            "pmid": "36184922",
            "title": "HA Filler for Nasolabial Fold Correction",
            "authors": "Park JH et al.",
            "year": 2022,
            "study_type": "RCT",
            "sample_size": 96,
            "conclusion": "HA 필러 팔자주름 개선, 효과 9개월 유지",
        }
    ],
    "mild pigmentation laser treatment necessity": [
        {
            "pmid": "35901234",
            "title": "Laser Toning for Mild Pigmentation",
            "authors": "Lee JW et al.",
            "year": 2022,
            "study_type": "Review",
            "sample_size": None,
            "conclusion": "경미한 색소침착(mild)에서 레이저 임상적 근거 부족. PIH 부작용 위험.",
        }
    ],
}


def _mock_search(query: str) -> list:
    q_lower = query.lower()
    for key, hits in _MOCK_PUBMED_RESULTS.items():
        key_tokens = key.split()
        if all(tok in q_lower for tok in key_tokens):
            return hits
    for key, hits in _MOCK_PUBMED_RESULTS.items():
        if any(tok in q_lower for tok in key.split()):
            return hits
    return []


def _esearch(query: str, max_results: int) -> list:
    url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list) -> list:
    if not pmids:
        return []
    url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return _parse_efetch_xml(resp.text)


def _parse_efetch_xml(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    parsed = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        year_el = article.find(".//PubDate/Year") or article.find(".//PubDate/MedlineDate")
        pub_type_els = article.findall(".//PublicationType")
        abstract_els = article.findall(".//Abstract/AbstractText")
        author_els = article.findall(".//AuthorList/Author")

        authors = []
        for a in author_els[:3]:
            last = a.findtext("LastName") or ""
            init = a.findtext("Initials") or ""
            name = f"{last} {init}".strip()
            if name:
                authors.append(name)
        authors_str = ", ".join(authors)
        if len(author_els) > 3:
            authors_str += " et al."

        year_text = year_el.text if year_el is not None and year_el.text else ""
        year = None
        if year_text:
            digits = "".join(ch for ch in year_text if ch.isdigit())
            if digits[:4].isdigit():
                year = int(digits[:4])

        study_type = pub_type_els[0].text if pub_type_els else None
        abstract = " ".join((el.text or "") for el in abstract_els).strip()

        parsed.append({
            "pmid": pmid_el.text if pmid_el is not None else None,
            "title": title_el.text if title_el is not None else None,
            "authors": authors_str or None,
            "year": year,
            "study_type": study_type,
            "sample_size": None,
            "conclusion": abstract[:600] if abstract else None,
        })
    return parsed


@tool
def search_pubmed(query: str, max_results: int = 5) -> dict:
    """PubMed에서 의학 논문을 검색합니다. 시술 근거 확인, 두 시술 비교, 적응증 확인 시 사용합니다. 시술을 추천하기 전에 반드시 호출하세요. 쿼리는 반드시 영문으로 작성하세요.

    Args:
        query: 영문 검색 쿼리. 예: 'forehead wrinkle severe botulinum toxin efficacy'.
        max_results: 반환할 논문 개수 (기본 5).

    Returns:
        results, total_found, query_used 키를 가진 dict.
    """
    if USE_MOCK_PUBMED:
        hits = _mock_search(query)[:max_results]
        return {
            "results": hits,
            "total_found": len(hits),
            "query_used": query,
        }

    try:
        pmids = _esearch(query, max_results)
        results = _efetch(pmids)
        return {
            "results": results,
            "total_found": len(results),
            "query_used": query,
        }
    except requests.RequestException as e:
        return {
            "results": [],
            "total_found": 0,
            "query_used": query,
            "error": f"PubMed API 호출 실패: {e}",
        }
