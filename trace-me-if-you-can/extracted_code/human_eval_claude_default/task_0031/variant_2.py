import math

return n >= 2 and all(n % k != 0 for k in range(2, int(math.sqrt(n)) + 1))
