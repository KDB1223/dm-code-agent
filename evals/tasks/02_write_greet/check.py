"""Golden check: greet returns Hello, <name>."""

import greet


def main() -> None:
    assert greet.greet("Ada") == "Hello, Ada"
    assert greet.greet("世界") == "Hello, 世界"
    print("golden ok")


if __name__ == "__main__":
    main()
