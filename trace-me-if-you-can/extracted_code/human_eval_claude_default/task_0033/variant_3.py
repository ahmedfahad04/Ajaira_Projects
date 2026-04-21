l = list(l)
step_size = 3
positions = []
elements = []
for i in range(0, len(l), step_size):
    positions.append(i)
    elements.append(l[i])
elements.sort()
for pos, elem in zip(positions, elements):
    l[pos] = elem
return l
