if not strings:
    return None

longest = strings[0]
for s in strings[1:]:
    if len(s) > len(longest):
        longest = s
return longest
