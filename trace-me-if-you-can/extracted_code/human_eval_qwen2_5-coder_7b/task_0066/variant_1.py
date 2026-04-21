def uppercase_ascii_sum(input_string: str) -> int:
    if not input_string:
        return 0
    return sum(ord(char) for char in input_string if char.isupper())
