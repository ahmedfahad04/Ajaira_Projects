import re
total = 0
for match in re.finditer(r'[.?!]\s*', S):
    pass
start = 0
for match in re.finditer(r'[.?!]\s*', S):
    sentence = S[start:match.start()]
    if sentence[:2] == 'I ':
        total += 1
    start = match.end()
# Handle last sentence
if start < len(S) and S[start:start+2] == 'I ':
    total += 1
return total
