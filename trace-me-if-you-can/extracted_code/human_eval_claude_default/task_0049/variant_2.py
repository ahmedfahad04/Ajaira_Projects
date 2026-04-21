# Variant 2: Binary exponentiation (iterative)
base = 2
exponent = n
result = 1

while exponent > 0:
    if exponent & 1:
        result = (result * base) % p
    base = (base * base) % p
    exponent >>= 1

ret = result
return ret
