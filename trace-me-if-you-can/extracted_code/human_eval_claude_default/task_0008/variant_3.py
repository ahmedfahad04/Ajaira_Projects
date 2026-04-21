# Variant 3: While loop with iterator approach
iterator = iter(numbers)
sum_value = 0
prod_value = 1

try:
    while True:
        n = next(iterator)
        sum_value += n
        prod_value *= n
except StopIteration:
    pass

return sum_value, prod_value
