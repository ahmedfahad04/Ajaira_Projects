# Variant 3: Accumulator pattern with early computation
total = 0
for token in s.split(' '):
    if token.isdigit():
        total += int(token)
return n - total
