def collect_substrings(text):
    substrings = []
    for idx in range(len(text)):
        substrings.append(text[:idx + 1])
    return substrings
