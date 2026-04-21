l = list(l)
third_elements = [l[i] for i in range(0, len(l), 3)]
third_elements.sort()
for i, val in enumerate(third_elements):
    l[i * 3] = val
return l
