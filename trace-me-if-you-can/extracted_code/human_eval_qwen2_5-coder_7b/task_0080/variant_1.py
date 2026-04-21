def validate_string(s: str) -> bool:
    if len(s) < 3:
        return False

    for i in range(len(s) - 2):
        if s[i] in [s[i+1], s[i+2]]:
            return False
    return True
