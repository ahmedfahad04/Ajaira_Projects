filtered = []
for value in values:
    if isinstance(value, int):
        filtered += [value]
return filtered
