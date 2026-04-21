def refactored_version_4(lst1, lst2):
    # Using min() with key parameter for direct comparison
    return min(lst1, lst2, key=lambda lst: sum(len(st) for st in lst))
