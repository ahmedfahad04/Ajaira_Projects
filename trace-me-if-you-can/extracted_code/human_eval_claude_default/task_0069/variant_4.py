freq_dict = {}
result = -1

for value in lst:
    freq_dict[value] = freq_dict.get(value, 0) + 1
    if freq_dict[value] >= value and value > result:
        result = value

return result
