total_length1 = sum(len(st) for st in lst1)
total_length2 = sum(len(st) for st in lst2)

if total_length1 <= total_length2:
    return lst1
else:
    return lst2
