current_max = float('-inf')
output_list = []

for number in numbers:
    if number > current_max:
        current_max = number
    output_list.append(current_max)

return output_list
