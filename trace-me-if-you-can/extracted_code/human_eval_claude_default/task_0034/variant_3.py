def remove_duplicates_and_sort(l):
    from collections import OrderedDict
    return sorted(OrderedDict.fromkeys(l).keys())
