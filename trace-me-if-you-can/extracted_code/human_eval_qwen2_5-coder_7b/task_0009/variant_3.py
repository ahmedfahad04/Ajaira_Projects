max_so_far = -float('inf')
output = []

for item in numbers:
    max_so_far = max(max_so_far, item)
    output.append(max_so_far)

return output
