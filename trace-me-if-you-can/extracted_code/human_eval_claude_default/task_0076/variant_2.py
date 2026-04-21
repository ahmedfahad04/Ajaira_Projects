import math

if n == 1:
    return x == 1
if n == 0 or x <= 0:
    return False
if x == 1:
    return True

log_result = math.log(x) / math.log(n)
rounded = round(log_result)
return abs(log_result - rounded) < 1e-10 and n ** rounded == x
