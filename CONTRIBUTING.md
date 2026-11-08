# Contributing

dm-agent is a small, readable kernel. Changes should stay that way.

## Setup

```bash
python -m pip install -e ".[dev]"
pytest
python -m evals.run
```

`pytest` covers the ReAct loop, the workspace jail, JSONL replay, and the offline eval. The eval does not call a model API.

## What to keep true

- File tools and `run_pytest` refuse paths that resolve outside the workspace, including symlink targets.
- `run_shell` stays a best-effort jail, not a sandbox. If you change `src/dm_agent/safety.py`, extend `tests/test_safety.py` and say so in the README safety section.
- Session logs stay append-only JSONL. Replay reads the log and does not re-run tools.
- New tools return structured text the model can read, and they have a unit test plus a line in both READMEs.

## Pull requests

Open a PR into `main` with a short description of the behavior change and how you ran `pytest`. Do not commit API keys or `.dm-agent/` session logs.
