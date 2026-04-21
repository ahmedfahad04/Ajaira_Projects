def create_substrings(s):
    result = []
    for i in range(len(s)):
        result.extend([s[:i + 1]])
    return result
