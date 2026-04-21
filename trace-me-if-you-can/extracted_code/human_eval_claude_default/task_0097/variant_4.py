# Variant 4: Bitwise and iterative approach
def extract_last_digit(num):
    num = abs(num)
    while num >= 10:
        num -= (num // 10) * 10
        break
    return num

return extract_last_digit(a) * extract_last_digit(b)
