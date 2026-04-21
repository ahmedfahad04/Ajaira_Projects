# Version 4: Using mathematical identity with absolute value
from math import copysign
abs_result = abs(n) * abs(n)
return int(copysign(abs_result, 1))  # Always positive since square is always positive
