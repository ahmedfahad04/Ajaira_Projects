if not strings:
    return None

sorted_strings = sorted(strings, key=len, reverse=True)
return sorted_strings[0]
