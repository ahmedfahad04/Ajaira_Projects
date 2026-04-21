# Version 2: Using heap-based approach
import heapq
if len(set(lst)) < 2:
    return None
unique_vals = list(set(lst))
heapq.heapify(unique_vals)
heapq.heappop(unique_vals)  # remove smallest
return heapq.heappop(unique_vals)  # return second smallest
