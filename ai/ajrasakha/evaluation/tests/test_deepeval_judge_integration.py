"""Opt-in integration test: makes REAL judge calls and spends money.

Runs only when RUN_JUDGE_INTEGRATION=1 and the judge is configured in
ai/.env (or the environment). Plain `pytest` skips it.
"""

import os

import pytest

# Imported first: this loads ai/.env, so the key is visible to the checks below.
import ajrasakha.evaluation.deepeval_metrics as dm

METRICS = [
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualRelevancyMetric",
]


def test_deepeval_judge_integration():
    if os.getenv("RUN_JUDGE_INTEGRATION") != "1":
        pytest.skip("Set RUN_JUDGE_INTEGRATION=1 to run this test (it spends money).")

    dm.get_judge_model.cache_clear()

    try:
        config = dm.load_judge_config()
    except dm.JudgeSetupError as exc:
        pytest.skip(f"Judge not configured: {exc}")

    # Label only: never print the key, the base URL or the config object.
    print(f"\nJudge: {config.provider}:{config.model}")

    question = "What is the current price of paddy in Mandya?"
    answer = "Paddy in Mandya is around Rs 2,300 per quintal at the mandi."
    context = ["[daily_price] Mandya paddy modal price is Rs 2,300 per quintal."]

    results = dm.evaluate_answer_with_deepeval(question, answer, context)

    for metric in METRICS:
        entry = results.get(metric, {})
        score = entry.get("score")
        reason = entry.get("reason")
        print(f"{metric}: {score}")

        assert score is not None, f"{metric} returned no score. Reason: {reason}"
        assert isinstance(score, (int, float)) and 0 <= score <= 1, (
            f"{metric} score {score!r} is out of range"
        )
        assert isinstance(reason, str) and reason.strip(), f"{metric} reason is empty"