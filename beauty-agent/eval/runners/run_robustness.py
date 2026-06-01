"""Robustness scenarios — graceful handling of error paths.

Each scenario asserts that the agent did not crash AND that the response contains
expected error-handling substrings.
"""
from __future__ import annotations

import time
import traceback

from eval.runners._common import load_dataset, save_result  # noqa: E402
from agent.graph import ChatSession  # noqa: E402


def _monkeypatch_pubmed_fail():
    import tools.search_pubmed as sp

    def _boom(*args, **kwargs):
        import requests
        raise requests.RequestException("monkeypatched failure")

    sp._esearch = _boom


def _run_scenario(sc: dict) -> dict:
    if sc.get("monkeypatch") == "pubmed_esearch_fail":
        _monkeypatch_pubmed_fail()

    session = ChatSession()
    t0 = time.time()
    output = ""
    error = None
    try:
        for pre in sc.get("preconditions", []):
            session.send(pre)
        output = session.send(sc["user"])
    except Exception as e:
        error = str(e)
        traceback.print_exc()
    elapsed = time.time() - t0

    expect = sc["expect"]
    crashed = error is not None
    msg = (output + " " + (session.final_answer or ""))

    # Substring checks (any-of)
    substrs = expect.get("graceful_message_substrings", [])
    any_hit = any(s in msg for s in substrs) if substrs else True

    tool_called = expect.get("tool_call_present")
    tool_present = (tool_called in [a.get("name") for a in (session.actions or [])]) if tool_called else None

    scores_ok = None
    age_none = None
    if expect.get("skin_scores_populated") is not None:
        scores_ok = bool(session.skin_scores and session.skin_scores.get("raw_scores"))
    if expect.get("age_is_none") is not None:
        scores_ok2 = session.skin_scores or {}
        age_none = (scores_ok2.get("age") is None)

    routed_insufficient = None
    if expect.get("routed_to_insufficient"):
        # The insufficient_response node injects a known phrase
        routed_insufficient = ("최종 레포트를 작성하려면" in (session.final_answer or ""))

    # Pass rules
    must_not_crash_ok = (not crashed) if expect.get("must_not_crash") else True
    checks = {
        "must_not_crash": must_not_crash_ok,
        "graceful_substring_present": any_hit,
        "tool_called_as_expected": (tool_present in (None, True)),
        "skin_scores_populated": (scores_ok in (None, True)) if expect.get("skin_scores_populated") else True,
        "age_is_none": (age_none in (None, True)) if expect.get("age_is_none") else True,
        "routed_to_insufficient": (routed_insufficient in (None, True)) if expect.get("routed_to_insufficient") else True,
    }
    passed = all(checks.values())

    return {
        "id": sc["id"],
        "user": sc["user"],
        "elapsed_sec": elapsed,
        "crashed": crashed,
        "error": error,
        "tools_called": [a.get("name") for a in (session.actions or [])],
        "checks": checks,
        "passed": passed,
        "message_preview": (msg or "")[:400],
    }


def main() -> dict:
    ds = load_dataset("robustness_scenarios.json")
    results = []
    t0 = time.time()
    for sc in ds["scenarios"]:
        print(f"[robust] {sc['id']} ...")
        results.append(_run_scenario(sc))
    elapsed = time.time() - t0
    n = len(results)
    pass_n = sum(1 for r in results if r["passed"])
    summary = {
        "metric": "robustness",
        "n": n,
        "pass": pass_n,
        "pass_rate": pass_n / n if n else 0.0,
        "elapsed_sec": elapsed,
        "scenarios": results,
    }
    save_result("robustness", summary)
    print(f"[robust] pass={pass_n}/{n} elapsed={elapsed:.1f}s")
    return summary


if __name__ == "__main__":
    main()
