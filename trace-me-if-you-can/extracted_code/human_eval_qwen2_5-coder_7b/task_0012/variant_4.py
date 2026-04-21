if not strings:
    return None

max_len = 0
for s in strings:
    if len(s) > max_len:
        max_len = len(s)
longest_str = next((s for s in strings if len(s) == max_len), None)
return longest_str
