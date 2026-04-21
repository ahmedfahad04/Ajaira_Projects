# Variant 4: Using enumerate and consuming iterator
return max((i for i, _ in enumerate(string, 1)), default=0)
