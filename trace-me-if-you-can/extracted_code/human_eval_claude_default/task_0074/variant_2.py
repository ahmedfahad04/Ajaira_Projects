def refactored_version_2(lst1, lst2):
    # Using map() and sum()
    l1 = sum(map(len, lst1))
    l2 = sum(map(len, lst2))
    
    if l1 <= l2:
        return lst1
    else:
        return lst2
