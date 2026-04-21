max_val = None
result_list = []

for value in numbers:
    if max_val is None or value > max_val:
        max_val = value
    result_list.append(max_val)

return result_list
