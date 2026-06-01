"""Combined image-driven runner — runs the full agent pipeline ONCE per image and
produces BOTH pubmed.json and db.json outputs.

Includes rate-limit backoff: retries 429 errors with sleep, and a configurable
inter-image cooldown to stay under TPM caps (gpt-4o = 30k TPM).

Per image:
  1) skin_analyze
  2) recommend_treatment_db
  3) search_pubmed
  4) final_report
Then judges:
  - per-article retrieval relevance (Claude)
  - final report faithfulness against retrieved abstracts (Claude)
  - per-treatment validity (Claude)
  - report patient-helpfulness (Claude)
And auto-computes:
  - PubMed citation correctness
  - DB rule vs vector_fallback ratio, age group consistency, code/PMID grounding,
    compress retention.
"""
from __future__ import annotations

import json
import time
import traceback

from eval.runners._common import load_dataset, save_result  # noqa: E402
from eval.runners.run_pubmed_eval import (
    _judge_relevance, _judge_faithfulness, _citation_check, _aggregate as _aggregate_pubmed,
)
from eval.runners.run_db_eval import (
    _judge_treatment, _judge_report, _age_group_consistent,
    _compress_coverage, _report_grounding, _aggregate as _aggregate_db,
)
from agent.graph import ChatSession  # noqa: E402


def _send_with_retry(session: ChatSession, text: str, max_retries: int = 4) -> None:
    """Retry on 429 (rate limit) with exponential-ish backoff parsed from message."""
    import re
    for attempt in range(max_retries):
        try:
            session.send(text)
            return
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" not in msg and "rate_limit" not in msg.lower():
                raise
            wait_match = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(wait_match.group(1)) + 2 if wait_match else 30 * (attempt + 1)
            print(f"  [retry {attempt + 1}/{max_retries}] 429; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
    # Last attempt: let exception propagate
    session.send(text)


def _run_image(image: dict, cooldown_sec: float = 0.0) -> dict:
    session = ChatSession()
    t0 = time.time()
    try:
        _send_with_retry(session, f"image_path={image['path']} {image['gender']} 진단해줘")
        _send_with_retry(session, "AuraDB에서 시술 추천해줘")
        _send_with_retry(session, "PubMed 논문 근거도 찾아줘")
        _send_with_retry(session, "최종 레포트 작성해줘")
    except Exception as e:
        return {"id": image["id"], "error": str(e), "trace": traceback.format_exc()}
    elapsed = time.time() - t0
    if cooldown_sec > 0:
        print(f"  cooldown {cooldown_sec:.0f}s", flush=True)
        time.sleep(cooldown_sec)

    skin_scores = session.skin_scores or {}
    age = skin_scores.get("age")
    db_recs = session.db_recommendations or []
    pubmed_recs = session.pubmed_recommendations or []
    report = session.final_answer or ""
    compressed = session._snapshot().get("compressed_summary") or ""

    # === PubMed metrics ===
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

    faith = _judge_faithfulness(report, pubmed_recs) if pubmed_recs else {"parsed": {"claims": []}, "raw": "", "error": None}
    cite = _citation_check(report, pubmed_recs)

    pubmed_entry = {
        "id": image["id"],
        "image_path": image["path"],
        "gender": image["gender"],
        "pipeline_elapsed_sec": elapsed,
        "iteration_count": session.iteration_count,
        "pubmed_recommendations": pubmed_recs,
        "final_report": report,
        "relevance": relevance_results,
        "faithfulness": faith,
        "citation_check": cite,
    }

    # === DB metrics ===
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

    treatment_judges = []
    for r in db_recs:
        j = _judge_treatment(r, age)
        treatment_judges.append({
            "region": r.get("region_ko"),
            "code": r.get("code"),
            "judge": j,
        })

    report_judge = _judge_report(report) if report else {"parsed": None, "raw": "", "error": "no_report"}

    db_entry = {
        "id": image["id"],
        "image_path": image["path"],
        "elapsed_sec": elapsed,
        "iteration_count": session.iteration_count,
        "age": age,
        "n_db_recs": len(db_recs),
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

    return {"pubmed": pubmed_entry, "db": db_entry, "elapsed": elapsed}


def main():
    import os
    cooldown = float(os.getenv("COMBINED_COOLDOWN_SEC", "60"))
    ds = load_dataset("eval_images.json")
    pubmed_results = []
    db_results = []
    t0 = time.time()
    for img in ds["images"]:
        print(f"[combined] running {img['id']} ...")
        out = _run_image(img, cooldown_sec=cooldown)
        if "error" in out:
            pubmed_results.append({"id": img["id"], "error": out["error"]})
            db_results.append({"id": img["id"], "error": out["error"]})
        else:
            pubmed_results.append(out["pubmed"])
            db_results.append(out["db"])
        print(f"  → done in {out.get('elapsed', 0):.1f}s")

    pubmed_summary = _aggregate_pubmed(pubmed_results)
    pubmed_summary["elapsed_sec"] = time.time() - t0
    save_result("pubmed", pubmed_summary)

    db_summary = _aggregate_db(db_results)
    db_summary["elapsed_sec"] = time.time() - t0
    save_result("db", db_summary)

    print(
        f"[combined] pubmed: relevance={pubmed_summary['retrieval']['avg_judge_score']} "
        f"fab={pubmed_summary['faithfulness']['fabrication_rate']} | "
        f"db: rule={db_summary['db']['rule_match_ratio']} "
        f"treat={db_summary['db']['treatment_validity_avg_0to2']} | "
        f"elapsed={pubmed_summary['elapsed_sec']:.1f}s"
    )


if __name__ == "__main__":
    main()
