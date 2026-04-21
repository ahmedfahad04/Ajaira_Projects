# Version 5: Array-based approach with enumerate
mean = sum(numbers) / len(numbers)
total_deviation = sum(abs(numbers[i] - mean) for i in range(len(numbers)))
return total_deviation / len(numbers)
