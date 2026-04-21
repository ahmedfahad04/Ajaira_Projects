import bisect
sorted_l = []
for item in l:
    bisect.insort(sorted_l, item)
mid = len(sorted_l) // 2
return sorted_l[mid] if len(sorted_l) % 2 == 1 else (sorted_l[mid - 1] + sorted_l[mid]) / 2.0
