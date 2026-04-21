intersection = set()
for first_list_element in l1:
    if first_list_element in l2:
        intersection.add(first_list_element)
return sorted(list(intersection))
