def remove_duplicates_and_sort(l):
    seen = set()
    unique_items = []
    for item in l:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return sorted(unique_items)
