def refactored_version_1(lst1, lst2):
    # Using sum() with generator expression
    l1 = sum(len(st) for st in lst1)
    l2 = sum(len(st) for st in lst2)
    
    return lst1 if l1 <= l2 else lst2
