total_sums = 0
total_products = 1

for element in numbers:
    total_sums += element
    total_products *= element

return total_sums, total_products
