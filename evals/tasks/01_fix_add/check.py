"""Golden check: add returns a sum."""

import math_utils


def main() -> None:
    assert math_utils.add(2, 3) == 5
    assert math_utils.add(-1, 1) == 0
    print("golden ok")


if __name__ == "__main__":
    main()
