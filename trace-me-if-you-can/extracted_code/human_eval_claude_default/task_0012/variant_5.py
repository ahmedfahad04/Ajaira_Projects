if not strings:
    return None

longest_length = 0
result = None
for s in strings:
    if len(s) > longest_length:
        longest_length = len(s)
        result = s
return result
