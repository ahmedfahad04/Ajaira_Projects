def uppercase_chars_ascii_sum(s: str) -> int:
    if s == "":
        return 0
    return sum(ord(ch) for ch in s if ch.isupper())
