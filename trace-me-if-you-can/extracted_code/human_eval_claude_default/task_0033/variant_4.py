import heapq
l = list(l)
third_vals = []
third_indices = []
for i in range(0, len(l), 3):
    heapq.heappush(third_vals, l[i])
    third_indices.append(i)
for i in third_indices:
    l[i] = heapq.heappop(third_vals)
return l
