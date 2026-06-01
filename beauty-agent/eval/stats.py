"""Statistical significance analysis for eval results.

All metrics here are computed from small samples (n=3–35), so we use
small-sample friendly methods:

- Proportions  → exact binomial test + Wilson score interval (Clopper–Pearson too)
- Ordinal scores (0-2, 1-5) → bootstrap CI for mean + Wilcoxon signed-rank vs midpoint
- 2×2 confusion matrix (per-class) → exact binomial CI

Each test reports H0, statistic, p-value, and 95% CI so the reader can judge.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from eval.runners._common import RESULTS_DIR  # noqa: E402


def _load(name: str):
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — works for small n and k=0 or k=n."""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial CI (more conservative than Wilson, recommended for n<20)."""
    if n == 0:
        return (0.0, 1.0)
    alpha = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def binom_test_better_than(k: int, n: int, p0: float) -> dict:
    """One-sided exact binomial test: H1: p > p0."""
    if n == 0:
        return {"p_value": 1.0, "k": k, "n": n, "p_hat": 0.0, "p0": p0}
    res = stats.binomtest(k, n, p=p0, alternative="greater")
    return {"p_value": float(res.pvalue), "k": k, "n": n, "p_hat": k / n, "p0": p0}


def binom_test_below(k: int, n: int, p0: float) -> dict:
    """One-sided exact binomial test: H1: p < p0."""
    if n == 0:
        return {"p_value": 1.0, "k": k, "n": n, "p_hat": 0.0, "p0": p0}
    res = stats.binomtest(k, n, p=p0, alternative="less")
    return {"p_value": float(res.pvalue), "k": k, "n": n, "p_hat": k / n, "p0": p0}


def bootstrap_mean_ci(values: list[float], conf: float = 0.95, n_boot: int = 10000, seed: int = 0):
    if not values:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.var() == 0:
        # All identical — CI is degenerate, mean is exact
        return {"mean": float(arr.mean()), "ci_low": float(arr.mean()), "ci_high": float(arr.mean()),
                "n": len(arr), "note": "degenerate (zero variance)"}
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, (1 - conf) / 2))
    hi = float(np.quantile(means, 1 - (1 - conf) / 2))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": len(arr)}


def wilcoxon_vs_midpoint(values: list[float], midpoint: float) -> dict | None:
    """One-sample Wilcoxon signed-rank test against H0: median = midpoint."""
    if not values or len(values) < 5:
        return None
    diffs = np.asarray(values) - midpoint
    if np.all(diffs == 0):
        return {"p_value": 1.0, "note": "all values equal to midpoint"}
    try:
        res = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "midpoint": midpoint, "n": len(values)}
    except ValueError as e:
        return {"error": str(e)}


def analyze_intent(d: dict) -> dict:
    n = d["total"]
    k = d["correct"]
    return {
        "name": "Intent classification accuracy",
        "n": n,
        "k": k,
        "p_hat": k / n,
        "wilson_95ci": wilson_ci(k, n),
        "clopper_pearson_95ci": clopper_pearson(k, n),
        "test_vs_random_0.5": binom_test_better_than(k, n, 0.5),
        "test_vs_majority_baseline": binom_test_better_than(
            k, n,
            # majority baseline = max class proportion in ground truth
            max(
                sum(1 for c in d["cases"] if c["label"] == "report"),
                sum(1 for c in d["cases"] if c["label"] == "general"),
            ) / n,
        ),
        "interpretation": (
            "H0: 분류기 정확도가 50% 무작위 추측과 같다. "
            f"binomial test p={binom_test_better_than(k, n, 0.5)['p_value']:.4g} → "
            f"{'유의 (p<0.05)' if binom_test_better_than(k, n, 0.5)['p_value'] < 0.05 else '비유의'}"
        ),
    }


def analyze_tool_calls(d: dict) -> dict:
    ok = [s for s in d["scenarios"] if not s.get("error")]
    n = len(ok)
    k_sel = sum(1 for s in ok if s["selection_correct"])
    k_guard = sum(1 for s in ok if s["guard_required"] and s["guard_pass"])
    n_guard = sum(1 for s in ok if s["guard_required"])
    n_chitchat = sum(1 for s in ok if s["expected_tools"] == [])

    return {
        "name": "Tool calling accuracy",
        "selection": {
            "n": n, "k": k_sel, "p_hat": k_sel / n,
            "wilson_95ci": wilson_ci(k_sel, n),
            "clopper_pearson_95ci": clopper_pearson(k_sel, n),
            "test_vs_random_0.25": binom_test_better_than(k_sel, n, 0.25),  # 4 possible tool sets baseline
        },
        "guard": {
            "n": n_guard, "k": k_guard, "p_hat": (k_guard / n_guard) if n_guard else None,
            "clopper_pearson_95ci": clopper_pearson(k_guard, n_guard) if n_guard else None,
            "note": f"가드 케이스 n={n_guard}만으로는 강한 결론 어려움. CI 폭으로 불확실성 표시.",
        },
        "interpretation": (
            f"selection {k_sel}/{n} → 95% CI [{wilson_ci(k_sel, n)[0]:.3f}, {wilson_ci(k_sel, n)[1]:.3f}]. "
            f"n={n}이 작아 CI가 넓다. random(0.25) 대비 p="
            f"{binom_test_better_than(k_sel, n, 0.25)['p_value']:.4g}."
        ),
    }


def analyze_pubmed(d: dict) -> dict:
    ret = d["retrieval"]
    faith = d["faithfulness"]

    # Relevance scores (0/1/2). Extract from cases.
    rel_scores: list[int] = []
    for img in d["images"]:
        for r in img.get("relevance", []):
            parsed = r["judge"].get("parsed")
            if parsed and "score" in parsed:
                rel_scores.append(int(parsed["score"]))

    # All claims labels
    labels = []
    for img in d["images"]:
        parsed = img["faithfulness"].get("parsed") or {}
        for cl in parsed.get("claims", []):
            labels.append(cl.get("label"))

    n_claims = len(labels)
    k_fab = sum(1 for l in labels if l == "fabricated")
    k_sup = sum(1 for l in labels if l == "supported")
    k_contra = sum(1 for l in labels if l == "contradicted")

    return {
        "name": "PubMed retrieval & faithfulness",
        "retrieval_relevance_score": {
            "n_articles": len(rel_scores),
            "mean": float(np.mean(rel_scores)) if rel_scores else None,
            "bootstrap_95ci": bootstrap_mean_ci(rel_scores),
            "wilcoxon_vs_midpoint_1.0": wilcoxon_vs_midpoint(rel_scores, 1.0),
            "note": (
                f"모든 {len(rel_scores)}개 article이 동일 score. 분산이 0이라 부트스트랩 CI는 점추정과 같다. "
                f"표본이 작아 score<max 케이스가 단순히 관찰되지 않은 것일 수 있다 "
                f"(true score<2 비율의 Wilson 95% upper bound: {wilson_ci(0, len(rel_scores))[1]:.3f})."
            ),
        },
        "fabrication_rate": {
            "n_claims": n_claims, "k_fabricated": k_fab,
            "p_hat": k_fab / n_claims if n_claims else None,
            "wilson_95ci": wilson_ci(k_fab, n_claims),
            "clopper_pearson_95ci": clopper_pearson(k_fab, n_claims),
            "test_below_10pct": binom_test_below(k_fab, n_claims, 0.10),
            "interpretation": (
                f"H0: 환각 비율 ≥10%. exact binomial p="
                f"{binom_test_below(k_fab, n_claims, 0.10)['p_value']:.4g} → "
                f"{'10% 미만이라는 증거 있음' if binom_test_below(k_fab, n_claims, 0.10)['p_value'] < 0.05 else '10% 미만이라는 증거 부족'}."
            ),
        },
        "supported_rate": {
            "n_claims": n_claims, "k_supported": k_sup,
            "p_hat": k_sup / n_claims if n_claims else None,
            "wilson_95ci": wilson_ci(k_sup, n_claims),
            "note": "abstract에 직접 근거가 있는 claim 비율. 나머지는 '의학적으로 타당하나 비근거' 또는 fabrication.",
        },
        "contradicted_rate": {
            "n_claims": n_claims, "k_contradicted": k_contra,
            "wilson_95ci": wilson_ci(k_contra, n_claims),
        },
    }


def analyze_db(d: dict) -> dict:
    db_block = d["db"]
    fr_block = d["final_report"]

    rule_k = db_block["rule_count"]
    rule_n = db_block["rule_count"] + db_block["vector_fallback_count"]

    # treatment validity scores
    tv_scores = []
    for img in d["images"]:
        for tj in img.get("treatment_judges", []):
            parsed = tj["judge"].get("parsed")
            if parsed and "score" in parsed:
                tv_scores.append(int(parsed["score"]))

    # Report axes (n=3 — too small for inference, descriptive only)
    rq_axes = {"clarity": [], "actionability": [], "safety_awareness": [], "overall": []}
    for img in d["images"]:
        parsed = img.get("report_judge", {}).get("parsed") or {}
        for k in rq_axes:
            if k in parsed:
                rq_axes[k].append(float(parsed[k]))

    return {
        "name": "DB recommendations & final report",
        "rule_match_ratio": {
            "n": rule_n, "k": rule_k,
            "p_hat": rule_k / rule_n if rule_n else None,
            "wilson_95ci": wilson_ci(rule_k, rule_n),
            "clopper_pearson_95ci": clopper_pearson(rule_k, rule_n),
            "test_above_50pct": binom_test_better_than(rule_k, rule_n, 0.50),
            "interpretation": (
                f"rule 매칭 {rule_k}/{rule_n} = {rule_k/rule_n:.3f}. "
                f"H0: rule 매칭률 ≤50% 대비 p="
                f"{binom_test_better_than(rule_k, rule_n, 0.50)['p_value']:.4g}."
            ),
        },
        "treatment_validity": {
            "n": len(tv_scores), "mean": float(np.mean(tv_scores)) if tv_scores else None,
            "bootstrap_95ci": bootstrap_mean_ci(tv_scores),
            "score_distribution": {str(s): tv_scores.count(s) for s in sorted(set(tv_scores))} if tv_scores else None,
            "fraction_lt_2_wilson_95ci": wilson_ci(
                sum(1 for s in tv_scores if s < 2), len(tv_scores)
            ) if tv_scores else None,
        },
        "report_axes": {
            "n_images": len(d["images"]),
            "raw_per_axis": rq_axes,
            "bootstrap_95ci_per_axis": {k: bootstrap_mean_ci(v) for k, v in rq_axes.items()},
            "wilcoxon_vs_midpoint_3_per_axis": {
                k: wilcoxon_vs_midpoint(v, 3.0) for k, v in rq_axes.items() if v
            },
            "warning": f"n={len(d['images'])} 이미지. 평균 차이 검정엔 검정력 부족하나 분포 자체는 의미 있음.",
        },
    }


def analyze_robustness(d: dict) -> dict:
    n = d["n"]
    k = d["pass"]
    return {
        "name": "Robustness",
        "n": n, "k": k, "p_hat": k / n,
        "clopper_pearson_95ci": clopper_pearson(k, n),
        "wilson_95ci": wilson_ci(k, n),
        "note": f"n={n}은 통계적 결론을 내기에 매우 작음. CI [{clopper_pearson(k, n)[0]:.2f}, {clopper_pearson(k, n)[1]:.2f}]는 거의 모든 값 포괄.",
    }


def power_warnings(sample_sizes: dict) -> list[str]:
    return [
        (
            f"**표본 크기**: intent n={sample_sizes['intent']}, tool_call n={sample_sizes['tools']}, "
            f"image-driven n_images={sample_sizes['images']} (PubMed n_articles={sample_sizes['pubmed_articles']}, "
            f"n_claims={sample_sizes['pubmed_claims']}; DB n_recommendations={sample_sizes['db_recs']}), "
            f"robustness n={sample_sizes['robustness']}."
        ),
        "proportion 메트릭은 exact binomial CI(Clopper–Pearson)로 표시; n이 큰 PubMed/DB 비율은 좁은 CI 확보.",
        "Treatment validity 같은 ordinal score는 bootstrap mean CI로 보고. 분산=0이면 \"불완전 케이스 미포착\"의 가능성.",
        "Report quality(n=12)는 평균 차이 검정에는 검정력 부족하나 점수 분포로 정성적 신호 해석 가능.",
        "의료 도메인이므로 fabrication, contradicted 1건도 운영상 중요. 95% CI 상한이 임상 허용치 이하인지를 함께 보라.",
    ]


def build():
    intent = _load("intent")
    tools = _load("tool_calls")
    pubmed = _load("pubmed")
    db = _load("db")
    robust = _load("robustness")

    sample_sizes = {
        "intent": intent["total"] if intent else 0,
        "tools": len(tools["scenarios"]) if tools else 0,
        "images": pubmed["n_images"] if pubmed else 0,
        "pubmed_articles": pubmed["retrieval"]["n_judged"] if pubmed else 0,
        "pubmed_claims": pubmed["faithfulness"]["total_claims"] if pubmed else 0,
        "db_recs": (db["db"]["rule_count"] + db["db"]["vector_fallback_count"]) if db else 0,
        "robustness": robust["n"] if robust else 0,
    }

    stats_out = {
        "sample_sizes": sample_sizes,
        "intent": analyze_intent(intent) if intent else None,
        "tool_calls": analyze_tool_calls(tools) if tools else None,
        "pubmed": analyze_pubmed(pubmed) if pubmed else None,
        "db": analyze_db(db) if db else None,
        "robustness": analyze_robustness(robust) if robust else None,
        "warnings": power_warnings(sample_sizes),
    }
    out_json = RESULTS_DIR / "STATS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2, default=str)

    md = render_markdown(stats_out)
    out_md = RESULTS_DIR / "STATS.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"wrote {out_json}\nwrote {out_md}")


def _fmt_ci(ci):
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt(v, fmt=".4g"):
    if v is None:
        return "n/a"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def _sig(p):
    if p is None:
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return " (n.s.)"


def render_markdown(s: dict) -> str:
    L: list[str] = ["# Beauty Agent 통계적 유의성 분석", ""]
    L.append("표본이 작아 모든 비율은 **exact binomial(Clopper–Pearson)** 95% CI를 보고하고, ")
    L.append("핵심 가설에는 **one-sided exact binomial test**, 점수형 메트릭은 **bootstrap 10k resamples** 평균 CI를 사용했다.")
    L.append("")
    L.append("- `*` p<0.05, `**` p<0.01, `***` p<0.001, `(n.s.)` 비유의")
    L.append("")

    # Intent
    i = s["intent"]
    if i:
        L.append("## 1. Intent classification")
        L.append(f"- n={i['n']}, accuracy = {i['k']}/{i['n']} = {i['p_hat']:.3f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(i['wilson_95ci'])}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(i['clopper_pearson_95ci'])}")
        t = i["test_vs_random_0.5"]
        L.append(f"- H1: accuracy > 50% (random): p = {t['p_value']:.4g}{_sig(t['p_value'])}")
        tm = i["test_vs_majority_baseline"]
        L.append(f"- H1: accuracy > majority baseline (p0={tm['p0']:.3f}): p = {tm['p_value']:.4g}{_sig(tm['p_value'])}")
        L.append(f"- 해석: {i['interpretation']}")
        L.append("")

    # Tools
    t = s["tool_calls"]
    if t:
        L.append("## 2. Tool calling accuracy")
        sel = t["selection"]
        L.append(f"### Selection accuracy")
        L.append(f"- n={sel['n']}, k={sel['k']}, p̂={sel['p_hat']:.3f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(sel['wilson_95ci'])}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(sel['clopper_pearson_95ci'])}")
        L.append(f"- H1: accuracy > 0.25 (random tool set): p = {sel['test_vs_random_0.25']['p_value']:.4g}{_sig(sel['test_vs_random_0.25']['p_value'])}")
        g = t["guard"]
        L.append(f"### Guard pass rate")
        L.append(f"- n={g['n']}, k={g['k']}, p̂={_fmt(g['p_hat'], '.3f')}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(g['clopper_pearson_95ci'])}")
        L.append(f"- ⚠ {g['note']}")
        L.append(f"- 해석: {t['interpretation']}")
        L.append("")

    # PubMed
    p = s["pubmed"]
    if p:
        L.append("## 3. PubMed retrieval & faithfulness")
        r = p["retrieval_relevance_score"]
        L.append(f"### Retrieval relevance (0–2 judge score)")
        L.append(f"- n={r['n_articles']}, mean={_fmt(r['mean'], '.3f')}")
        ci = r["bootstrap_95ci"]
        if ci:
            L.append(f"- bootstrap 95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
            if ci.get("note"):
                L.append(f"  - {ci['note']}")
        w = r["wilcoxon_vs_midpoint_1.0"]
        if w:
            if "p_value" in w:
                L.append(f"- Wilcoxon vs midpoint=1.0 (one-sided): p = {w['p_value']:.4g}{_sig(w['p_value'])}")
            elif "note" in w:
                L.append(f"- Wilcoxon: {w['note']}")
        L.append(f"- ⚠ {r['note']}")
        L.append("")

        f_ = p["fabrication_rate"]
        L.append(f"### Fabrication rate (PubMed hallucination)")
        L.append(f"- n_claims={f_['n_claims']}, fabricated={f_['k_fabricated']}, p̂={f_['p_hat']:.4f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(f_['wilson_95ci'])}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(f_['clopper_pearson_95ci'])}")
        tb = f_["test_below_10pct"]
        L.append(f"- H1: fabrication rate < 10%: exact binomial p = {tb['p_value']:.4g}{_sig(tb['p_value'])}")
        L.append(f"- 해석: {f_['interpretation']}")
        L.append("")

        sr = p["supported_rate"]
        L.append(f"### Supported rate")
        L.append(f"- n_claims={sr['n_claims']}, supported={sr['k_supported']}, p̂={sr['p_hat']:.3f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(sr['wilson_95ci'])}")
        L.append(f"- {sr['note']}")
        L.append("")

        cr = p["contradicted_rate"]
        L.append(f"### Contradicted rate")
        L.append(f"- n_claims={cr['n_claims']}, contradicted={cr['k_contradicted']}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(cr['wilson_95ci'])}")
        L.append("")

    # DB
    d = s["db"]
    if d:
        L.append("## 4. DB recommendations & final report")
        rm = d["rule_match_ratio"]
        L.append(f"### Rule match ratio")
        L.append(f"- n={rm['n']}, k={rm['k']}, p̂={rm['p_hat']:.3f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(rm['wilson_95ci'])}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(rm['clopper_pearson_95ci'])}")
        tba = rm["test_above_50pct"]
        L.append(f"- H1: rule_ratio > 50%: p = {tba['p_value']:.4g}{_sig(tba['p_value'])}")
        L.append(f"- 해석: {rm['interpretation']}")
        L.append("")

        tv = d["treatment_validity"]
        L.append(f"### Treatment validity (0–2)")
        L.append(f"- n={tv['n']}, mean={_fmt(tv['mean'], '.3f')}")
        ci = tv["bootstrap_95ci"]
        if ci:
            L.append(f"- bootstrap 95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        if tv.get("score_distribution"):
            L.append(f"- score 분포: {tv['score_distribution']}")
        if tv.get("fraction_lt_2_wilson_95ci"):
            cii = tv["fraction_lt_2_wilson_95ci"]
            L.append(f"- score<2 비율 Wilson 95% CI: [{cii[0]:.3f}, {cii[1]:.3f}]")
        L.append("")

        ra = d["report_axes"]
        L.append(f"### Report quality (1–5)")
        L.append(f"- n_images = {ra['n_images']}")
        L.append("")
        L.append("| axis | mean | bootstrap mean 95% CI | Wilcoxon vs midpoint=3.0 |")
        L.append("|---|---|---|---|")
        for axis, vals in ra["raw_per_axis"].items():
            cii = ra["bootstrap_95ci_per_axis"].get(axis)
            ci_str = f"[{cii['ci_low']:.2f}, {cii['ci_high']:.2f}]" if cii else "n/a"
            mean = sum(vals)/len(vals) if vals else None
            w = ra["wilcoxon_vs_midpoint_3_per_axis"].get(axis) if ra.get("wilcoxon_vs_midpoint_3_per_axis") else None
            if w and "p_value" in w:
                w_str = f"p={w['p_value']:.4g}{_sig(w['p_value'])}"
            elif w and "note" in w:
                w_str = w["note"]
            else:
                w_str = "n/a"
            L.append(f"| {axis} | {_fmt(mean, '.2f')} | {ci_str} | {w_str} |")
        L.append("")
        L.append(f"- ⚠ {ra['warning']}")
        L.append("")

    # Robustness
    r = s["robustness"]
    if r:
        L.append("## 5. Robustness")
        L.append(f"- n={r['n']}, k={r['k']}, p̂={r['p_hat']:.3f}")
        L.append(f"- Wilson 95% CI: {_fmt_ci(r['wilson_95ci'])}")
        L.append(f"- Clopper–Pearson 95% CI: {_fmt_ci(r['clopper_pearson_95ci'])}")
        L.append(f"- ⚠ {r['note']}")
        L.append("")

    L.append("## 한계 (검정력)")
    for w in s["warnings"]:
        L.append(f"- {w[2:] if w.startswith('- ') else w}")
    return "\n".join(L)


if __name__ == "__main__":
    build()
