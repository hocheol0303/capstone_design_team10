"""Aggregate results/*.json into a single human-readable REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

from eval.runners._common import RESULTS_DIR  # noqa: E402


def _load(name: str):
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, fmt=".3f"):
    if v is None:
        return "n/a"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def build_report() -> str:
    intent = _load("intent")
    tools = _load("tool_calls")
    pubmed = _load("pubmed")
    db = _load("db")
    robust = _load("robustness")

    lines: list[str] = ["# Beauty Agent 성능 평가 리포트", ""]
    lines.append("- judge model: `claude-sonnet-4-6`")
    lines.append("- agent model: `gpt-4o-mini` (temperature=0)")
    lines.append("")

    # 1. Intent
    lines.append("## 1. Intent Classification")
    if intent:
        lines.append(f"- accuracy: **{_fmt(intent['accuracy'])}** ({intent['correct']}/{intent['total']})")
        lines.append(f"- elapsed: {intent['elapsed_sec']:.1f}s")
        lines.append("")
        lines.append("| class | precision | recall | f1 |")
        lines.append("|---|---|---|---|")
        for cls, m in intent["per_class"].items():
            lines.append(f"| {cls} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} |")
        lines.append("")
        wrong = [c for c in intent["cases"] if not c["correct"]]
        if wrong:
            lines.append("**오분류**:")
            for w in wrong:
                lines.append(f"- `{w['id']}` text={w['text']!r} label={w['label']} pred={w['pred']}")
        lines.append("")
    else:
        lines.append("_(결과 없음)_\n")

    # 2. Tool Calls
    lines.append("## 2. Tool Calling Accuracy")
    if tools:
        lines.append(f"- selection accuracy: **{_fmt(tools['selection_accuracy'])}**")
        lines.append(f"- order accuracy (multi-tool only): {_fmt(tools['order_accuracy'])}")
        lines.append(f"- argument accuracy: {_fmt(tools['argument_accuracy'])}")
        lines.append(f"- guard pass rate: {_fmt(tools['guard_pass_rate'])}")
        lines.append(f"- avg iteration count: {_fmt(tools['avg_iteration_count'], '.2f')}")
        lines.append(f"- elapsed: {tools['elapsed_sec']:.1f}s")
        lines.append("")
        lines.append("| id | expected | actual | selection | order | args | guard |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in tools["scenarios"]:
            if s.get("error"):
                lines.append(f"| {s['id']} | – | ERROR: {s['error'][:60]} | – | – | – | – |")
                continue
            lines.append(
                f"| {s['id']} | `{s['expected_tools']}` | `{s['actual_tools']}` | "
                f"{'OK' if s['selection_correct'] else 'FAIL'} | "
                f"{'OK' if s['order_correct'] else 'FAIL'} | "
                f"{'OK' if s['argument_correct'] else 'FAIL'} | "
                f"{'OK' if s['guard_pass'] else 'FAIL'} |"
            )
        lines.append("")
    else:
        lines.append("_(결과 없음)_\n")

    # 3. PubMed
    lines.append("## 3. PubMed Retrieval + Faithfulness")
    if pubmed:
        ret = pubmed["retrieval"]
        faith = pubmed["faithfulness"]
        sim = ret.get("embedding_similarity_stats") or {}
        lines.append(f"- 이미지 수: {pubmed['n_images']}, 채점된 articles: {ret['n_judged']}")
        lines.append(f"- 평균 judge 점수 (0–2): **{_fmt(ret['avg_judge_score'])}**")
        lines.append(f"- 관련성 ≥1 비율: {_fmt(ret['relevance_at_least_1_rate'])}")
        if sim:
            lines.append(
                f"- 임베딩 유사도: mean={_fmt(sim['mean'])}, "
                f"min={_fmt(sim['min'])}, p25={_fmt(sim['p25'])}, "
                f"median={_fmt(sim['median'])}, p75={_fmt(sim['p75'])}, max={_fmt(sim['max'])}"
            )
        lines.append("")
        lines.append("**Faithfulness (claim 분해 + abstract 대조)**")
        lines.append(f"- total claims: {faith['total_claims']}")
        lines.append(f"- label counts: {faith['label_counts']}")
        lines.append(f"- supported rate: **{_fmt(faith['supported_rate'])}**")
        lines.append(f"- fabrication rate (PubMed hallucination): **{_fmt(faith['fabrication_rate'])}**")
        lines.append("")
        bad = pubmed["citation"]["images_with_invalid_citations"]
        if bad:
            lines.append(f"- ⚠ citation 불일치 이미지: {bad}")
        else:
            lines.append("- citation: 전 이미지 일치 (PMID 인용은 검색 결과와 매칭)")
        lines.append(f"- elapsed: {pubmed['elapsed_sec']:.1f}s")
        lines.append("")
    else:
        lines.append("_(결과 없음)_\n")

    # 4. DB + Report
    lines.append("## 4. DB Recommendation + Final Report Quality")
    if db:
        d = db["db"]
        fr = db["final_report"]
        lines.append("**DB 추천**")
        lines.append(f"- 평균 region 커버리지 (10개 중): {_fmt(d['avg_coverage_of_10_regions'])}")
        lines.append(f"- rule 매칭 비율: **{_fmt(d['rule_match_ratio'])}** (rule={d['rule_count']}, vector_fallback={d['vector_fallback_count']})")
        lines.append(f"- age_group 정합성: {_fmt(d['age_group_consistency_rate'])}")
        lines.append(f"- treatment validity (0–2 judge): **{_fmt(d['treatment_validity_avg_0to2'])}**")
        lines.append("")
        lines.append("**최종 레포트 품질 (1–5 judge)**")
        lines.append(f"- clarity: {_fmt(fr['avg_clarity_1to5'])}")
        lines.append(f"- actionability: {_fmt(fr['avg_actionability_1to5'])}")
        lines.append(f"- safety_awareness: {_fmt(fr['avg_safety_awareness_1to5'])}")
        lines.append(f"- overall: **{_fmt(fr['avg_overall_1to5'])}**")
        lines.append("")
        lines.append("**Compress 정보 보존**")
        lines.append(f"- treatment code 보존율: {_fmt(fr['compress_code_retention_avg'])}")
        lines.append(f"- PMID 보존율: {_fmt(fr['compress_pmid_retention_avg'])}")
        lines.append("")
        lines.append("**Report grounding (인용 검증)**")
        lines.append(f"- code grounding rate: {_fmt(fr['code_grounding_rate'])}")
        lines.append(f"- PMID grounding rate: {_fmt(fr['pmid_grounding_rate'])}")
        lines.append(f"- elapsed: {db['elapsed_sec']:.1f}s")
        lines.append("")
    else:
        lines.append("_(결과 없음)_\n")

    # 5. Robustness
    lines.append("## 5. Robustness")
    if robust:
        lines.append(f"- pass rate: **{_fmt(robust['pass_rate'])}** ({robust['pass']}/{robust['n']})")
        lines.append(f"- elapsed: {robust['elapsed_sec']:.1f}s")
        lines.append("")
        for s in robust["scenarios"]:
            checks = ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in s["checks"].items())
            mark = "PASS" if s["passed"] else "FAIL"
            lines.append(f"- **{mark}** `{s['id']}`: {checks}")
            if not s["passed"] or s["crashed"]:
                lines.append(f"  - msg preview: {s['message_preview'][:200]}")
        lines.append("")
    else:
        lines.append("_(결과 없음)_\n")

    # Header summary
    head = ["", "## 요약 (한눈에)", "", "| 차원 | 핵심 지표 | 값 |", "|---|---|---|"]
    if intent:
        head.append(f"| Intent | accuracy | {_fmt(intent['accuracy'])} |")
    if tools:
        head.append(f"| Tool selection | accuracy | {_fmt(tools['selection_accuracy'])} |")
        head.append(f"| Tool guard | pass rate | {_fmt(tools['guard_pass_rate'])} |")
    if pubmed:
        head.append(f"| PubMed relevance | judge 0–2 avg | {_fmt(pubmed['retrieval']['avg_judge_score'])} |")
        head.append(f"| PubMed hallucination | fabrication rate | {_fmt(pubmed['faithfulness']['fabrication_rate'])} |")
    if db:
        head.append(f"| DB rule match | rule_ratio | {_fmt(db['db']['rule_match_ratio'])} |")
        head.append(f"| DB treatment | judge 0–2 avg | {_fmt(db['db']['treatment_validity_avg_0to2'])} |")
        head.append(f"| Final report | overall 1–5 | {_fmt(db['final_report']['avg_overall_1to5'])} |")
    if robust:
        head.append(f"| Robustness | pass rate | {_fmt(robust['pass_rate'])} |")
    head.append("")

    return "\n".join(lines[:2] + head + lines[2:])


def main():
    text = build_report()
    out = RESULTS_DIR / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
