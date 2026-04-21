# Version 2: Functional approach using map and lambda
mean = sum(numbers) / len(numbers)
deviations = map(lambda x: abs(x - mean), numbers)
return sum(deviations) / len(numbers)
