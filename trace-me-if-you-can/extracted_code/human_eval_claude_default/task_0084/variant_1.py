digit_sum = 0
temp = N
while temp > 0:
    digit_sum += temp % 10
    temp //= 10
return format(digit_sum, 'b')
