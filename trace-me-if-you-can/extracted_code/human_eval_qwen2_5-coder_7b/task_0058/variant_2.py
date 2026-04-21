common = set()
for element in l1:
    if element in l2:
        common.add(element)
return sorted(list(common))
