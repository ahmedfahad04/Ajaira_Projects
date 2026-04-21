sum_lengths = lambda lst: sum(len(st) for st in lst)

lengths1 = sum_lengths(lst1)
lengths2 = sum_lengths(lst2)

return lst1 if lengths1 <= lengths2 else lst2
