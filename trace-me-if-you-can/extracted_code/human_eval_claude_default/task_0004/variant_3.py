# Version 3: Single-pass with accumulator pattern
mean = sum(numbers) / len(numbers)
deviation_sum = 0
for value in numbers:
    deviation_sum += abs(value - mean)
return deviation_sum / len(numbers)
