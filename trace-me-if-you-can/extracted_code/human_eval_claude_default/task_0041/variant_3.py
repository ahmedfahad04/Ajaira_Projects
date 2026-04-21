# Version 3: Using iterative approach (repeated addition)
if n == 0:
    return 0
result = 0
abs_n = abs(n)
for i in range(abs_n):
    result += abs_n
return result
