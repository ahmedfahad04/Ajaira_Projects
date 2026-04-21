def calc_uppercase_ascii_total(s: str) -> int:
    if s == "":
        return 0
    return sum(ord(c) for c in s if c.isupper())
