ret = set()
l2_set = set(l2)
for e1 in l1:
    if e1 in l2_set:
        ret.add(e1)
return sorted(list(ret))
