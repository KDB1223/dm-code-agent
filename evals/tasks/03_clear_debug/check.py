"""Golden check: missing files return an empty string, and the debug hook is gone."""

from pathlib import Path

import app


def main() -> None:
    assert app.load_text("missing-file.txt") == ""
    source = Path("app.py").read_text(encoding="utf-8")
    assert "breakpoint" not in source
    assert "except:" not in source
    print("golden ok")


if __name__ == "__main__":
    main()
