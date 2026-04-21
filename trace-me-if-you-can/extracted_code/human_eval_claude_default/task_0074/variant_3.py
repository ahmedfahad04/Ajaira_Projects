def refactored_version_3(lst1, lst2):
    # Using reduce from functools
    from functools import reduce
    
    l1 = reduce(lambda acc, st: acc + len(st), lst1, 0)
    l2 = reduce(lambda acc, st: acc + len(st), lst2, 0)
    
    return lst1 if l1 <= l2 else lst2
