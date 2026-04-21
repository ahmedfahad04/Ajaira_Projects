# Version 1: Two-pass approach with explicit intermediate variables
total = sum(numbers)
count = len(numbers)
mean = total / count
absolute_deviations = [abs(x - mean) for x in numbers]
return sum(absolute_deviations) / count
