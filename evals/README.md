# Offline evals

`python -m evals.run` copies each task into a temp directory, runs the real ReAct loop against the scripted turns in `task.json`, checks the JSONL log for the expected successful tool results, then runs `check.py`. No API key.

```bash
python -m evals.run
python -m evals.run --transcript fix-add
```

A full run's stdout matches `expected_output.txt`:

```text
fix-add        PASS
write-greet    PASS
clear-debug    PASS
score 3/3
```

| Task | What the script does | Golden check |
| --- | --- | --- |
| `fix-add` | Read, change `a - b` to `a + b`, lint | `add(2, 3) == 5` |
| `write-greet` | Implement `greet`, run pytest | `greet("Ada") == "Hello, Ada"` |
| `clear-debug` | Lint, remove bare `except` and `breakpoint()`, lint again | Missing file returns `""` and the debug call is gone |

The harness is deterministic on purpose, so the score can be graded without a model. A live model is the `dm-agent run` path in the README.
