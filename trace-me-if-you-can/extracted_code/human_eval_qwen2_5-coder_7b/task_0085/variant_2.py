total_sum = 0
for index in range(1, len(lst), 2):
    element = lst[index]
    if element % 2 == 0:
        total_sum += element
return total_sum
