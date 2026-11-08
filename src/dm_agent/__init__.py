"""Local-first coding agent with an append-only audit log.

The kernel is a ReAct loop: the model plans, calls a workspace tool, reads the
observation, and repeats until it gives a final answer or hits the step limit.
Every event is appended to a JSONL session file that can be inspected and
replayed without running the tools again.
"""

__version__ = "0.1.0"
