def compute_total_length(lst):
    return sum(len(st) for st in lst)

length_lst1 = compute_total_length(lst1)
length_lst2 = compute_total_length(lst2)

return lst1 if length_lst1 <= length_lst2 else lst2
