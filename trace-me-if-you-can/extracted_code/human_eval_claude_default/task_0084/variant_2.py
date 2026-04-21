# Variant 2: Recursive digit extraction
def extract_digits(n):
    if n == 0:
        return 0
    return (n % 10) + extract_digits(n // 10)

return bin(extract_digits(N))[2:]
