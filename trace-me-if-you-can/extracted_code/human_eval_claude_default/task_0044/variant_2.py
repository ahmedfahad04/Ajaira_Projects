digits = []
temp_x = x
while temp_x > 0:
    digits.append(str(temp_x % base))
    temp_x //= base
ret = ''.join(reversed(digits))
return ret
