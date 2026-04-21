import math

if n < 2:
    return False
limit = int(math.sqrt(n)) + 1
for k in range(2, limit):
    if n % k == 0:
        return False
return True
