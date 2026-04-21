# Variant 2: Filter and map functional approach
digits = filter(str.isdigit, s.split(' '))
return n - sum(map(int, digits))
