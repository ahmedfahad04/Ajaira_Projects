result = s
for _ in range(2):
    result = encode_cyclic(result)
return result
