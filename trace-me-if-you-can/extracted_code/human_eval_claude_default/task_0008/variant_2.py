# Variant 2: List comprehension with unpacking
sum_value = sum([n for n in numbers])
prod_value = 1
for n in numbers:
    prod_value *= n
return sum_value, prod_value
