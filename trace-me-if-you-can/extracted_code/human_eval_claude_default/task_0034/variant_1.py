def remove_duplicates_and_sort(l):
    result = []
    for item in l:
        if item not in result:
            result.append(item)
    result.sort()
    return result
