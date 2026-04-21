def refactored_version_5(lst1, lst2):
    # Using list comprehension and sum()
    l1 = sum([len(st) for st in lst1])
    l2 = sum([len(st) for st in lst2])
    
    return [lst1, lst2][l1 > l2]
