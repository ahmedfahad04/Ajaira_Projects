l = list(l)
indices = list(range(0, len(l), 3))
values = [l[i] for i in indices]
values.sort()
for idx, val in zip(indices, values):
    l[idx] = val
return l
