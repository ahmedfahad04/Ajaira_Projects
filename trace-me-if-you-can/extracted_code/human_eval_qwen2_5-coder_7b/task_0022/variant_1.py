filtered_values = []
for value in values:
    if isinstance(value, int):
        filtered_values.append(value)
return filtered_values
