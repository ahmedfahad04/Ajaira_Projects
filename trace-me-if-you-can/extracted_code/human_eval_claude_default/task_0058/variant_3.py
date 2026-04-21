l2_set = set(l2)
return sorted(list(set(filter(lambda x: x in l2_set, l1))))
