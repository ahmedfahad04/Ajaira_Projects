# Version 2: Using bit shifting for powers of 2 optimization with fallback
import math
if n != 0 and (n & (n - 1)) == 0:  # Check if n is power of 2
    log_n = int(math.log2(abs(n)))
    return (1 << (2 * log_n)) if n > 0 else (1 << (2 * log_n))
else:
    return n * n
