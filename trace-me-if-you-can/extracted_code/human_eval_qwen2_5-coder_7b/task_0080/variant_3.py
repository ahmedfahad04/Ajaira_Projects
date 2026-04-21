def is_valid_sequence(s: str) -> bool:
    if len(s) < 3:
        return False

    for index in range(len(s) - 2):
        if s[index] == s[index+1] or s[index+1] == s[index+2] or s[index] == s[index+2]:
            return False
    return True
