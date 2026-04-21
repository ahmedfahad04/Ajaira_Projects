digit_sum = sum([int(digit) for digit in str(N)])
binary_digits = []
if digit_sum == 0:
    return "0"
while digit_sum > 0:
    binary_digits.append(str(digit_sum & 1))
    digit_sum >>= 1
return ''.join(reversed(binary_digits))
