"""PubMed retrieval relevance + faithfulness.

For each eval image:
  1) Run skin_analyze (via ChatSession)
  2) Run search_pubmed
  3) Run a final_report (request "최종 레포트")
  4) Judge each (query → article) for relevance (0/1/2)
  5) Judge final report claims for faithfulness against the retrieved abstracts
  6) Cross-check citations (PMID/authors/year) against retrieved records
"""
from __future__ import annotations

import json
import time
import traceback

from eval.runners._common import load_dataset, save_result  # noqa: E402
from agent.graph import ChatSession  # noqa: E402
from eval.judges import claude_judge  # noqa: E402


RELEVANCE_SYSTEM = (
    "You are an expert dermatology research librarian. Given a body region and the PubMed query used to "
    "search for procedural evidence for that region, judge how relevant a retrieved article is to clinical "
    "treatment of that condition. Return JSON only.\n"
    "Schema: {\"score\": 0|1|2, \"reason\": \"short explanation\"}\n"
    "Scoring: 2 = directly relevant procedural/treatment evidence; 1 = related (epidemiology, pathophysiology, "
    "or adjacent treatment); 0 = unrelated."
)

FAITHFULNESS_SYSTEM = (
    "You are an evidence auditor for a Korean medical report citing PubMed abstracts. Decompose the report into "
    "individual factual claims that reference a paper (procedural efficacy, study finding, statistical claim, "
    "guideline). For each claim, classify against the supplied abstracts.\n"
    "Return JSON ONLY: {\"claims\": [{\"claim\": str, \"label\": \"supported\"|\"unsupported_but_plausible\"|\"contradicted\"|\"fabricated\", "
    "\"evidence_pmid\": str|null, \"note\": str}]}\n"
    "Definitions: supported = clearly stated in an abstract; unsupported_but_plausible = absent from abstracts "
    "but medically reasonable; contradicted = directly contradicts an abstract; fabricated = invented detail "
    "(authors, numbers, PMID, study type) not present in abstracts.\n"
    "ONLY consider abstracts provided as evidence. If the report makes a claim with no abstract evidence, label "
    "it unsupported_but_plausible or fabricated based on specificity."
)


def _judge_relevance(region_ko: str, query: str, title: str, abstract: str) -> dict:
    payload = json.dumps({
        "region": region_ko,
        "pubmed_query": query,
        "article_title": title,
        "article_abstract": abstract[:1500],
    }, ensure_ascii=False)
    return claude_judge.score(RELEVANCE_SYSTEM, payload, max_tokens=400)


def _judge_faithfulness(report: str, pubmed_recs: list[dict]) -> dict:
    payload = json.dumps({
        "final_report": report,
        "available_abstracts": [
            {"pmid": r.get("pmid"), "title": r.get("title"), "authors": r.get("authors"),
             "year": r.get("year"), "abstract": (r.get("abstract") or "")[:1500]}
            for r in pubmed_recs
        ],
    }, ensure_ascii=False)
    return claude_judge.score(FAITHFULNESS_SYSTEM, payload, max_tokens=2500)


def _citation_check(report: str, pubmed_recs: list[dict]) -> dict:
    """Direct substring check: any PMID/year mentioned in report must exist in pubmed_recs."""
    import re
    valid_pmids = {str(r.get("pmid")) for r in pubmed_recs if r.get("pmid")}
    mentioned = set(re.findall(r"PMID[:\s]*(\d{5,9})", report))
    extra = mentioned - valid_pmids
    return {
        "report_pmids": sorted(mentioned),
        "valid_pmids": sorted(valid_pmids),
        "unsupported_pmids": sorted(extra),
        "all_citations_valid": len(extra) == 0,
    }


def _run_image(image: dict) -> dict:
    session = ChatSession()
    t0 = time.time()
    try:
        session.send(f"image_path={image['path']} {image['gender']} 진단해줘")
        session.send("이제 PubMed 논문 근거를 찾아줘")
        session.send("최종 레포트 작성해줘")
    except Exception as e:
        return {"id": image["id"], "error": str(e), "trace": traceback.format_exc()}
    pipeline_elapsed = time.time() - t0

    pubmed_recs = session.pubmed_recommendations or []
    report = session.final_answer or ""

    # 1) Per-article relevance judgments
    relevance_results = []
    for rec in pubmed_recs:
        j = _judge_relevance(
            region_ko=rec.get("region_ko", ""),
            query=rec.get("query_used", ""),
            title=rec.get("title", "") or "",
            abstract=rec.get("abstract", "") or "",
        )
        relevance_results.append({
            "region": rec.get("region_ko"),
            "pmid": rec.get("pmid"),
            "query": rec.get("query_used"),
            "embedding_similarity": rec.get("similarity"),
            "judge": j,
        })

    # 2) Faithfulness of final report
    faith = _judge_faithfulness(report, pubmed_recs) if pubmed_recs else {"parsed": {"claims": []}, "raw": "", "error": None}

    # 3) Citation correctness
    cite = _citation_check(report, pubmed_recs)

    return {
        "id": image["id"],
        "image_path": image["path"],
        "gender": image["gender"],
        "pipeline_elapsed_sec": pipeline_elapsed,
        "iteration_count": session.iteration_count,
        "pubmed_recommendations": pubmed_recs,
        "final_report": report,
        "relevance": relevance_results,
        "faithfulness": faith,
        "citation_check": cite,
    }


def _aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]

    # Relevance
    rel_scores = []
    for r in ok:
        for item in r["relevance"]:
            parsed = item["judge"].get("parsed")
            if parsed and isinstance(parsed, dict) and "score" in parsed:
                rel_scores.append(parsed["score"])
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else None
    rel_at_least_1 = sum(1 for s in rel_scores if s >= 1) / len(rel_scores) if rel_scores else None

    # Faithfulness — fabrication rate
    label_counts = {"supported": 0, "unsupported_but_plausible": 0, "contradicted": 0, "fabricated": 0}
    total_claims = 0
    for r in ok:
        parsed = r["faithfulness"].get("parsed") or {}
        for cl in parsed.get("claims", []):
            lbl = cl.get("label")
            if lbl in label_counts:
                label_counts[lbl] += 1
                total_claims += 1
    fab_rate = label_counts["fabricated"] / total_claims if total_claims else None
    sup_rate = label_counts["supported"] / total_claims if total_claims else None

    # Citation
    invalid_cite_imgs = [r["id"] for r in ok if not r["citation_check"]["all_citations_valid"]]

    # Similarity stats
    sims = [item["embedding_similarity"] for r in ok for item in r["relevance"] if item["embedding_similarity"] is not None]
    if sims:
        sims_sorted = sorted(sims)
        sim_stats = {
            "mean": sum(sims) / len(sims),
            "min": sims_sorted[0],
            "p25": sims_sorted[len(sims) // 4],
            "median": sims_sorted[len(sims) // 2],
            "p75": sims_sorted[3 * len(sims) // 4],
            "max": sims_sorted[-1],
            "n": len(sims),
        }
    else:
        sim_stats = None

    return {
        "metric": "pubmed_eval",
        "n_images": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "retrieval": {
            "avg_judge_score": avg_rel,
            "relevance_at_least_1_rate": rel_at_least_1,
            "n_judged": len(rel_scores),
            "embedding_similarity_stats": sim_stats,
        },
        "faithfulness": {
            "total_claims": total_claims,
            "label_counts": label_counts,
            "fabrication_rate": fab_rate,
            "supported_rate": sup_rate,
        },
        "citation": {
            "images_with_invalid_citations": invalid_cite_imgs,
        },
        "images": results,
    }


def main() -> dict:
    ds = load_dataset("eval_images.json")
    results = []
    t0 = time.time()
    for img in ds["images"]:
        print(f"[pubmed] running {img['id']} ...")
        results.append(_run_image(img))
    summary = _aggregate(results)
    summary["elapsed_sec"] = time.time() - t0
    save_result("pubmed", summary)
    print(f"[pubmed] avg_relevance={summary['retrieval']['avg_judge_score']} "
          f"fabrication_rate={summary['faithfulness']['fabrication_rate']} "
          f"elapsed={summary['elapsed_sec']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
