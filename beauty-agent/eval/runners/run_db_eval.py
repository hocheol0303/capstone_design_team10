"""DB recommendation correctness + Final report quality.

For each eval image:
  1) Run skin_analyze → recommend_treatment_db → search_pubmed → final_report
  2) Collect db_recommendations + final report
  3) Judge treatment validity per region (Claude)
  4) Judge final report helpfulness (Claude, 5-point)
  5) Automatic: rule vs vector ratio, age group consistency, fact-coverage in
     compressed_summary, fuzzy match of report → recs
"""
from __future__ import annotations

import json
import re
import time
import traceback

from eval.runners._common import load_dataset, save_result  # noqa: E402
from agent.graph import ChatSession  # noqa: E402
from eval.judges import claude_judge  # noqa: E402


TREATMENT_VALIDITY_SYSTEM = (
    "You are a board-certified Korean dermatologist auditing treatment recommendations from a DB-driven agent. "
    "Given a body region, its diagnosis score (0-100, lower is worse), patient age, and a recommended "
    "treatment, judge if the recommendation is clinically appropriate. Return JSON only.\n"
    "Schema: {\"score\": 0|1|2, \"reason\": \"short\"}\n"
    "2 = clinically standard for this region/severity/age; "
    "1 = plausible but not first-line; "
    "0 = inappropriate or mismatched (wrong region, wrong severity)."
)

REPORT_QUALITY_SYSTEM = (
    "You are a patient-experience reviewer evaluating a Korean cosmetic dermatology report. Score the report "
    "on three axes (1-5 scale each). Return JSON only.\n"
    "Schema: {\"clarity\": 1-5, \"actionability\": 1-5, \"safety_awareness\": 1-5, \"overall\": 1-5, \"notes\": \"short\"}\n"
    "clarity: structure + readability for a non-specialist;\n"
    "actionability: are recommended next steps concrete?;\n"
    "safety_awareness: does it mention contraindications, side-effects, or consultation prompts?;\n"
    "overall: holistic patient-facing quality."
)


def _judge_treatment(rec: dict, age: float | None) -> dict:
    payload = json.dumps({
        "region_ko": rec.get("region_ko"),
        "score_0_100": rec.get("score"),
        "patient_age": age,
        "feature_name": rec.get("feature_name"),
        "code": rec.get("code"),
        "treatment_text": rec.get("treatment"),
        "customer_desc": (rec.get("customer_desc") or "")[:600],
    }, ensure_ascii=False)
    return claude_judge.score(TREATMENT_VALIDITY_SYSTEM, payload, max_tokens=300)


def _judge_report(report: str) -> dict:
    return claude_judge.score(REPORT_QUALITY_SYSTEM, report[:6000], max_tokens=400)


def _parse_age_group_bounds(s: str) -> tuple[int, int]:
    s = (s or "").strip()
    m = re.match(r"(\d+)\s*세\s*이하", s)
    if m:
        return (0, int(m.group(1)))
    m = re.match(r"(\d+)\s*-\s*(\d+)\s*세", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"(\d+)\s*세\s*이상", s)
    if m:
        return (int(m.group(1)), 999)
    return (-1, -1)


def _age_group_consistent(age: float | None, group: str | None) -> bool | None:
    if age is None or not group:
        return None
    lo, hi = _parse_age_group_bounds(group)
    if lo < 0:
        return None
    return lo <= int(round(age)) <= hi


def _compress_coverage(compressed: str, db_recs: list[dict], pubmed_recs: list[dict]) -> dict:
    """Fraction of (treatment code, PMID) tokens preserved in compressed summary."""
    compressed = compressed or ""
    codes = [r.get("code") for r in db_recs if r.get("code")]
    pmids = [r.get("pmid") for r in pubmed_recs if r.get("pmid")]
    codes_kept = sum(1 for c in codes if c and c in compressed)
    pmids_kept = sum(1 for p in pmids if p and p in compressed)
    return {
        "n_codes": len(codes),
        "codes_kept": codes_kept,
        "code_retention": codes_kept / len(codes) if codes else None,
        "n_pmids": len(pmids),
        "pmids_kept": pmids_kept,
        "pmid_retention": pmids_kept / len(pmids) if pmids else None,
    }


def _report_grounding(report: str, db_recs: list[dict], pubmed_recs: list[dict]) -> dict:
    """Each treatment code mentioned in report should appear in db_recs; each PMID in pubmed_recs."""
    report = report or ""
    valid_codes = {r.get("code") for r in db_recs if r.get("code")}
    valid_pmids = {str(r.get("pmid")) for r in pubmed_recs if r.get("pmid")}
    code_pattern = re.compile(r"\b([A-Z]{1,4}_\d+|[A-Z]{1,5}\d{2,4})\b")
    cited_codes = set(code_pattern.findall(report))
    cited_pmids = set(re.findall(r"PMID[:\s]*(\d{5,9})", report))
    return {
        "report_codes": sorted(cited_codes),
        "invalid_codes": sorted(cited_codes - valid_codes),
        "report_pmids": sorted(cited_pmids),
        "invalid_pmids": sorted(cited_pmids - valid_pmids),
        "code_grounded": len(cited_codes - valid_codes) == 0,
        "pmid_grounded": len(cited_pmids - valid_pmids) == 0,
    }


def _run_image(image: dict) -> dict:
    session = ChatSession()
    t0 = time.time()
    try:
        session.send(f"image_path={image['path']} {image['gender']} 진단해줘")
        session.send("AuraDB에서 시술 추천해줘")
        session.send("PubMed 논문 근거도 찾아줘")
        session.send("최종 레포트 작성해줘")
    except Exception as e:
        return {"id": image["id"], "error": str(e), "trace": traceback.format_exc()}
    elapsed = time.time() - t0

    skin_scores = session.skin_scores or {}
    age = skin_scores.get("age")
    db_recs = session.db_recommendations or []
    pubmed_recs = session.pubmed_recommendations or []
    report = session.final_answer or ""
    compressed = session._snapshot().get("compressed_summary") or ""

    # Automatic metrics
    n = len(db_recs)
    matched_by_counts = {"rule": 0, "vector_fallback": 0, "other": 0}
    age_group_consistent_count = 0
    age_group_total = 0
    for r in db_recs:
        mb = r.get("matched_by") or "other"
        matched_by_counts[mb] = matched_by_counts.get(mb, 0) + 1
        ok = _age_group_consistent(age, r.get("matched_age_group"))
        if ok is not None:
            age_group_total += 1
            if ok:
                age_group_consistent_count += 1

    compress = _compress_coverage(compressed, db_recs, pubmed_recs)
    grounding = _report_grounding(report, db_recs, pubmed_recs)

    # Judge treatments
    treatment_judges = []
    for r in db_recs:
        j = _judge_treatment(r, age)
        treatment_judges.append({
            "region": r.get("region_ko"),
            "code": r.get("code"),
            "judge": j,
        })

    report_judge = _judge_report(report) if report else {"parsed": None, "raw": "", "error": "no_report"}

    return {
        "id": image["id"],
        "image_path": image["path"],
        "elapsed_sec": elapsed,
        "iteration_count": session.iteration_count,
        "age": age,
        "n_db_recs": n,
        "matched_by_counts": matched_by_counts,
        "age_group_consistency": {
            "n_checked": age_group_total,
            "n_consistent": age_group_consistent_count,
            "rate": (age_group_consistent_count / age_group_total) if age_group_total else None,
        },
        "compress_coverage": compress,
        "report_grounding": grounding,
        "db_recommendations": db_recs,
        "final_report": report,
        "treatment_judges": treatment_judges,
        "report_judge": report_judge,
    }


def _aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]

    # Coverage: db recs / 10 regions
    coverage = [r["n_db_recs"] / 10.0 for r in ok]
    avg_coverage = sum(coverage) / len(coverage) if coverage else None

    # Matched_by aggregate
    rule_total = sum(r["matched_by_counts"].get("rule", 0) for r in ok)
    vec_total = sum(r["matched_by_counts"].get("vector_fallback", 0) for r in ok)
    db_total = rule_total + vec_total
    rule_ratio = rule_total / db_total if db_total else None

    # Age group consistency
    ag_n = sum(r["age_group_consistency"]["n_checked"] for r in ok)
    ag_c = sum(r["age_group_consistency"]["n_consistent"] for r in ok)
    ag_rate = ag_c / ag_n if ag_n else None

    # Treatment judge avg
    judge_scores = []
    for r in ok:
        for tj in r["treatment_judges"]:
            parsed = tj["judge"].get("parsed")
            if parsed and isinstance(parsed, dict) and "score" in parsed:
                judge_scores.append(parsed["score"])
    avg_treatment = sum(judge_scores) / len(judge_scores) if judge_scores else None

    # Report judge avg
    rq_axes = {"clarity": [], "actionability": [], "safety_awareness": [], "overall": []}
    for r in ok:
        parsed = r["report_judge"].get("parsed") or {}
        for k in rq_axes:
            if k in parsed:
                try:
                    rq_axes[k].append(float(parsed[k]))
                except (TypeError, ValueError):
                    pass
    rq_avg = {k: (sum(v) / len(v) if v else None) for k, v in rq_axes.items()}

    # Compress coverage
    code_retention = [r["compress_coverage"]["code_retention"] for r in ok if r["compress_coverage"]["code_retention"] is not None]
    pmid_retention = [r["compress_coverage"]["pmid_retention"] for r in ok if r["compress_coverage"]["pmid_retention"] is not None]

    # Grounding
    code_grounded = sum(1 for r in ok if r["report_grounding"]["code_grounded"])
    pmid_grounded = sum(1 for r in ok if r["report_grounding"]["pmid_grounded"])

    return {
        "metric": "db_eval",
        "n_images": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "db": {
            "avg_coverage_of_10_regions": avg_coverage,
            "rule_match_ratio": rule_ratio,
            "rule_count": rule_total,
            "vector_fallback_count": vec_total,
            "age_group_consistency_rate": ag_rate,
            "treatment_validity_avg_0to2": avg_treatment,
        },
        "final_report": {
            "avg_clarity_1to5": rq_avg["clarity"],
            "avg_actionability_1to5": rq_avg["actionability"],
            "avg_safety_awareness_1to5": rq_avg["safety_awareness"],
            "avg_overall_1to5": rq_avg["overall"],
            "compress_code_retention_avg": sum(code_retention) / len(code_retention) if code_retention else None,
            "compress_pmid_retention_avg": sum(pmid_retention) / len(pmid_retention) if pmid_retention else None,
            "code_grounding_rate": code_grounded / len(ok) if ok else None,
            "pmid_grounding_rate": pmid_grounded / len(ok) if ok else None,
        },
        "images": results,
    }


def main() -> dict:
    ds = load_dataset("eval_images.json")
    results = []
    t0 = time.time()
    for img in ds["images"]:
        print(f"[db] running {img['id']} ...")
        results.append(_run_image(img))
    summary = _aggregate(results)
    summary["elapsed_sec"] = time.time() - t0
    save_result("db", summary)
    print(f"[db] coverage={summary['db']['avg_coverage_of_10_regions']} "
          f"rule_ratio={summary['db']['rule_match_ratio']} "
          f"treatment_avg={summary['db']['treatment_validity_avg_0to2']} "
          f"report_overall={summary['final_report']['avg_overall_1to5']} "
          f"elapsed={summary['elapsed_sec']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
