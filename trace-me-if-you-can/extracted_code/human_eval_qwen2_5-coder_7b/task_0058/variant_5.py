common_items = set()
for item in l1:
    if item in l2:
        common_items.add(item)
return sorted(list(common_items))
