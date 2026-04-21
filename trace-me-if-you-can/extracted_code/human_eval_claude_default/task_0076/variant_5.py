if n == 1:
    return x == 1
if n == 0:
    return False

temp = x
while temp > 1:
    if temp % n != 0:
        return False
    temp //= n

return temp == 1
