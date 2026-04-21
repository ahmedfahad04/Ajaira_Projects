def sum_ascii_values_uppercase(s: str) -> int:
    if s == "":
        return 0
    return sum(ord(char) if char.isupper() else 0 for char in s)
