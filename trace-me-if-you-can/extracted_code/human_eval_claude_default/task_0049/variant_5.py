# Variant 5: Unrolled loop with batching
ret = 1
full_iterations = n // 4
remainder = n % 4

# Process 4 multiplications at once
for _ in range(full_iterations):
    ret = (ret * 16) % p  # 2^4 = 16

# Handle remaining iterations
for _ in range(remainder):
    ret = (ret * 2) % p

return ret
