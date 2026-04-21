# Version 1: Using enumerate and list comprehension
sorted_evens = sorted(l[::2])
result = []
even_idx = 0
for i in range(len(l)):
    if i % 2 == 0:
        result.append(sorted_evens[even_idx])
        even_idx += 1
    else:
        result.append(l[i])
return result
