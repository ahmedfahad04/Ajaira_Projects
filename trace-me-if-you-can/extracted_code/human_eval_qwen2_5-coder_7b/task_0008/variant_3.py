accumulator_sum = 0
accumulator_product = 1

for digit in numbers:
    accumulator_sum += digit
    accumulator_product *= digit

return accumulator_sum, accumulator_product
