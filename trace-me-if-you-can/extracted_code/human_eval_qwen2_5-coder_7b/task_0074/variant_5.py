def total_length(lst):
    total = 0
    for st in lst:
        total += len(st)
    return total

length1 = total_length(lst1)
length2 = total_length(lst2)

if length1 <= length2:
    return lst1
else:
    return lst2
