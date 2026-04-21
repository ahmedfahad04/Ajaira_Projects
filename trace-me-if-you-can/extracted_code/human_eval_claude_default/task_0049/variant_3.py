ret = 1
remaining = n
power_of_two = 2 % p

while remaining:
    if remaining & 1:
        ret = (ret * power_of_two) % p
    power_of_two = (power_of_two * power_of_two) % p
    remaining >>= 1

return ret
