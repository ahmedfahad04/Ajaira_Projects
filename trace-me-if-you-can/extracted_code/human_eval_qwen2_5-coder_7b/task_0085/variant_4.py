even_sum = 0
for index in range(1, len(lst), 2):
    if lst[index] % 2 == 0:
        even_sum += lst[index]
return even_sum
