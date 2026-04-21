calculate_sum = lambda lst: sum(map(len, lst))

sum1 = calculate_sum(lst1)
sum2 = calculate_sum(lst2)

if sum1 <= sum2:
    return lst1
else:
    return lst2
