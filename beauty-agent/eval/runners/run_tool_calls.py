"""Tool call accuracy — drives ChatSession through scenarios and compares actual
tool_call sequence against expected.

Metrics:
- tool_selection_accuracy: actual set == expected set
- order_correctness:       actual ordered prefix matches expected (multi-tool only)
- argument_correctness:    skin_analyze args match expected_args
- guard_pass_rate:         for guard scenarios, forbidden_tools were NOT called
"""
from __future__ import annotations

import time
import traceback

from eval.runners._common import load_dataset, save_result  # noqa: E402
from agent.graph import ChatSession  # noqa: E402


def _collect_tool_calls(session: ChatSession) -> list[dict]:
    """Read session.actions (list of tool_call dicts). actions accumulates across turns."""
    return list(session.actions or [])


def _send_with_retry(session: ChatSession, text: str, max_retries: int = 3) -> str:
    import re
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return session.send(text)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e)
            if "429" not in msg and "rate_limit" not in msg.lower():
                raise
            wait_match = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(wait_match.group(1)) + 2 if wait_match else 30 * (attempt + 1)
            print(f"  [retry {attempt + 1}/{max_retries}] 429; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last_exc  # type: ignore


def _run_scenario(scenario: dict) -> dict:
    session = ChatSession()
    pre_action_count = 0

    # Replay preconditions silently
    for pre in scenario.get("preconditions", []):
        try:
            _send_with_retry(session, pre)
        except Exception as e:
            return {"id": scenario["id"], "error": f"precondition failed: {e}", "trace": traceback.format_exc()}
        pre_action_count = len(_collect_tool_calls(session))

    t0 = time.time()
    try:
        output = _send_with_retry(session, scenario["user"])
    except Exception as e:
        return {"id": scenario["id"], "error": str(e), "trace": traceback.format_exc()}
    elapsed = time.time() - t0

    all_actions = _collect_tool_calls(session)
    new_actions = all_actions[pre_action_count:]
    actual_tools = [a.get("name") for a in new_actions]
    actual_args = {a.get("name"): a.get("args", {}) for a in new_actions}

    expected = scenario.get("expected_tools", [])
    forbidden = scenario.get("forbidden_tools", [])

    # Selection: set comparison
    selection_correct = set(actual_tools) == set(expected)
    # Order: only meaningful if expected has >1 tool
    if len(expected) > 1:
        # actual must contain expected in order (allow extras)
        order_correct = _is_subsequence(expected, actual_tools)
    else:
        order_correct = selection_correct

    # Argument check
    arg_results = {}
    for tool, exp_args in scenario.get("expected_args", {}).items():
        got = actual_args.get(tool, {})
        arg_results[tool] = {
            "expected": exp_args,
            "actual": got,
            "match": all(got.get(k) == v for k, v in exp_args.items()),
        }
    arg_correct = all(r["match"] for r in arg_results.values()) if arg_results else True

    # Guard: forbidden tools must not appear
    guard_pass = not any(t in forbidden for t in actual_tools) if forbidden else True

    iter_count = session.iteration_count
    final_answer = session.final_answer or output[:500]

    return {
        "id": scenario["id"],
        "user": scenario["user"],
        "expected_tools": expected,
        "actual_tools": actual_tools,
        "selection_correct": selection_correct,
        "order_correct": order_correct,
        "argument_correctness": arg_results,
        "argument_correct": arg_correct,
        "guard_required": scenario.get("guard", False),
        "guard_pass": guard_pass,
        "forbidden_tools": forbidden,
        "iteration_count": iter_count,
        "elapsed_sec": elapsed,
        "final_answer_preview": final_answer[:300],
    }


def _is_subsequence(needle: list, haystack: list) -> bool:
    it = iter(haystack)
    return all(any(h == n for h in it) for n in needle)


def main() -> dict:
    ds = load_dataset("tool_call_scenarios.json")
    scenarios = ds["scenarios"]

    results = []
    t0 = time.time()
    for sc in scenarios:
        print(f"[tool] running {sc['id']} ...")
        results.append(_run_scenario(sc))
    elapsed = time.time() - t0

    # Aggregate
    n = len(results)
    n_err = sum(1 for r in results if r.get("error"))
    ok = [r for r in results if not r.get("error")]

    selection_acc = sum(1 for r in ok if r["selection_correct"]) / len(ok) if ok else 0.0
    multi = [r for r in ok if len(r["expected_tools"]) > 1]
    order_acc = sum(1 for r in multi if r["order_correct"]) / len(multi) if multi else None
    with_args = [r for r in ok if r["argument_correctness"]]
    arg_acc = sum(1 for r in with_args if r["argument_correct"]) / len(with_args) if with_args else None
    guard_cases = [r for r in ok if r["guard_required"]]
    guard_acc = sum(1 for r in guard_cases if r["guard_pass"]) / len(guard_cases) if guard_cases else None

    avg_iter = sum(r.get("iteration_count", 0) for r in ok) / len(ok) if ok else 0.0

    summary = {
        "metric": "tool_call_accuracy",
        "total": n,
        "errors": n_err,
        "selection_accuracy": selection_acc,
        "order_accuracy": order_acc,
        "argument_accuracy": arg_acc,
        "guard_pass_rate": guard_acc,
        "avg_iteration_count": avg_iter,
        "elapsed_sec": elapsed,
        "scenarios": results,
    }
    save_result("tool_calls", summary)
    print(
        f"[tool] selection={selection_acc:.3f} order={order_acc} args={arg_acc} guard={guard_acc} "
        f"avg_iter={avg_iter:.2f} elapsed={elapsed:.1f}s"
    )
    return summary


if __name__ == "__main__":
    main()
