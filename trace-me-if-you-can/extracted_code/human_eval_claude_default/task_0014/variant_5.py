def generate_prefixes(string):
    result = []
    current_prefix = ""
    for char in string:
        current_prefix += char
        result.append(current_prefix)
    return result
