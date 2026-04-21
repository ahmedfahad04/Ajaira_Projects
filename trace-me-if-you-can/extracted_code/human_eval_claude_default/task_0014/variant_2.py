def generate_prefixes(string):
    result = []
    for i, _ in enumerate(string):
        result.append(string[:i+1])
    return result
