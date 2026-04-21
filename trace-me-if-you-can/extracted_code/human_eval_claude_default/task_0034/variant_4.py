def remove_duplicates_and_sort(l):
    import heapq
    unique_set = set(l)
    result = []
    heap = list(unique_set)
    heapq.heapify(heap)
    while heap:
        result.append(heapq.heappop(heap))
    return result
