current_highest = None
output_array = []

for digit in numbers:
    if current_highest is None or digit > current_highest:
        current_highest = digit
    output_array.append(current_highest)

return output_array
