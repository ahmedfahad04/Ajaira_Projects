# Version 1: Early termination with square root optimization
import math
for i in range(2, int(math.sqrt(n)) + 1):
    if n % i == 0:
        return n // i
return 1
