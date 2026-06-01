"""Intent classification accuracy.

Calls agent.helpers.classify_intent() directly (no graph needed).
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from eval.runners._common import load_dataset, save_result  # noqa: E402
from agent.helpers import classify_intent  # noqa: E402


def main() -> dict:
    ds = load_dataset("intent_cases.json")
    cases = ds["cases"]

    results = []
    t0 = time.time()
    for c in cases:
        pred = classify_intent(c["text"])
        results.append({"id": c["id"], "text": c["text"], "label": c["label"], "pred": pred, "correct": pred == c["label"]})

    elapsed = time.time() - t0

    # Aggregate
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["label"]][r["pred"]] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in ("report", "general"):
        tp = confusion[label][label]
        fn = sum(v for k, v in confusion[label].items() if k != label)
        fp = sum(confusion[other][label] for other in confusion if other != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fn": fn, "fp": fp}

    summary = {
        "metric": "intent_classification_accuracy",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "elapsed_sec": elapsed,
        "cases": results,
    }
    save_result("intent", summary)
    print(f"[intent] accuracy={accuracy:.3f} ({correct}/{total}) elapsed={elapsed:.1f}s")
    return summary


if __name__ == "__main__":
    main()
