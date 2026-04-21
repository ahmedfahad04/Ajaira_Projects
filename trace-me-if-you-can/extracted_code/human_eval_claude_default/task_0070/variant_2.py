def refactored_version_2(lst):
    """Using heap-based approach with min and max heaps"""
    import heapq
    
    if not lst:
        return []
    
    min_heap = lst.copy()
    max_heap = [-x for x in lst]
    heapq.heapify(min_heap)
    heapq.heapify(max_heap)
    
    res = []
    used = set()
    is_min_turn = True
    
    while len(res) < len(lst):
        if is_min_turn:
            while min_heap[0] in used:
                heapq.heappop(min_heap)
            val = heapq.heappop(min_heap)
        else:
            while -max_heap[0] in used:
                heapq.heappop(max_heap)
            val = -heapq.heappop(max_heap)
        
        res.append(val)
        used.add(val)
        is_min_turn = not is_min_turn
    
    return res
