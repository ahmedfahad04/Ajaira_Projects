if x == 0:
    return ""

# Calculate number of digits first
temp = x
digit_count = 0
while temp > 0:
    digit_count += 1
    temp //= base

# Build result using positional calculation
ret = ['0'] * digit_count
temp_x = x
for i in range(digit_count - 1, -1, -1):
    ret[i] = str(temp_x % base)
    temp_x //= base

return ''.join(ret)
