def check_divisor(n, k, limit):
    if k > limit:
        return True
    if n % k == 0:
        return False
    return check_divisor(n, k + 1, limit)

import math
if n < 2:
    return False
return check_divisor(n, 2, int(math.sqrt(n)))
