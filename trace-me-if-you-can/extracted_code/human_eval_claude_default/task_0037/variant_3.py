# Version 3: In-place reconstruction with deque
from collections import deque
sorted_evens = deque(sorted(l[::2]))
result = []
for i in range(len(l)):
    if i % 2 == 0:
        result.append(sorted_evens.popleft())
    else:
        result.append(l[i])
return result
