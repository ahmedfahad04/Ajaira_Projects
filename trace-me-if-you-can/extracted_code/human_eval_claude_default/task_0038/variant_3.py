# Variant 3: Loop-based approach for multiple applications
result = s
for _ in range(2):
    result = encode_cyclic(result)
return result
