# Variant 1: List comprehension approach
return n - sum(int(i) for i in s.split(' ') if i.isdigit())
