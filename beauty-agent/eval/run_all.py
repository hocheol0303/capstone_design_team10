"""End-to-end driver. Runs all dimensions and writes REPORT.md."""
from __future__ import annotations

from eval.runners import run_intent, run_tool_calls, run_pubmed_eval, run_db_eval, run_robustness  # noqa: E402
from eval import report  # noqa: E402


def main():
    run_intent.main()
    run_tool_calls.main()
    run_pubmed_eval.main()
    run_db_eval.main()
    run_robustness.main()
    report.main()


if __name__ == "__main__":
    main()
