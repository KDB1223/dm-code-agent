def load_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except:
        breakpoint()
        return ""
