# Variant 5: Using enumerate with tuple unpacking
sum_value, prod_value = 0, 1

for i, n in enumerate(numbers):
    sum_value += n
    prod_value *= n

return sum_value, prod_value
