import heapq
n = len(l)
if n % 2 == 1:
    return heapq.nsmallest(n // 2 + 1, l)[-1]
else:
    mid_elements = heapq.nsmallest(n // 2 + 1, l)[-2:]
    return sum(mid_elements) / 2.0
