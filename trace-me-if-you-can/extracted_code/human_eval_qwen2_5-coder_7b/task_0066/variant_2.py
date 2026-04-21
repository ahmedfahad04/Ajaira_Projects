def sum_uppercase_ascii_values(text: str) -> int:
    if len(text) == 0:
        return 0
    return sum(ord(ch) for ch in text if ch.isupper())
