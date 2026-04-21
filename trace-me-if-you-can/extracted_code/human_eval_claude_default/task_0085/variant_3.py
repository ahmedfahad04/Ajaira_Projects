even_sum = 0
i = 1
while i < len(lst):
    if lst[i] % 2 == 0:
        even_sum += lst[i]
    i += 2
return even_sum
