# Variant 3: Using divmod for digit extraction
total = 0
num = N
while num:
    num, digit = divmod(num, 10)
    total += digit
result = ""
if total == 0:
    return "0"
while total:
    result = str(total % 2) + result
    total //= 2
return result
