"""The offline eval suite is part of the reproducible demo."""

from pathlib import Path

from evals.run import TaskResult, format_summary, main


def test_summary_matches_the_checked_in_scoreboard() -> None:
    text = format_summary(
        [
            TaskResult("fix-add", True, "", ""),
            TaskResult("write-greet", True, "", ""),
            TaskResult("clear-debug", True, "", ""),
        ]
    )
    expected = Path(__file__).resolve().parents[1] / "evals" / "expected_output.txt"
    assert text == expected.read_text(encoding="utf-8")


def test_offline_eval_scores_three_of_three() -> None:
    assert main([]) == 0
