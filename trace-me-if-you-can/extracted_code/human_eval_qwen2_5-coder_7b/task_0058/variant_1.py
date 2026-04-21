def intersect_lists(l1, l2):
    common_elements = set()
    for item1 in l1:
        if item1 in l2:
            common_elements.add(item1)
    return sorted(list(common_elements))
