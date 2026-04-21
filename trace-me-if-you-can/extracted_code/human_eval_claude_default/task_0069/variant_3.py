# Variant 3: Set-based filtering with max function
frequency_map = {}
for element in lst:
    frequency_map[element] = frequency_map.get(element, 0) + 1

valid_numbers = [num for num, count in frequency_map.items() if count >= num]
return max(valid_numbers) if valid_numbers else -1
