def filter_strings(strings, substring):
    result = []
    for string in strings:
        if substring in string:
            result.append(string)
    return result
