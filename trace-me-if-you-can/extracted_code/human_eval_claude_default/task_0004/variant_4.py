# Version 4: Using statistics module approach with manual calculation
from statistics import mean as stat_mean
calculated_mean = stat_mean(numbers)
return sum(abs(num - calculated_mean) for num in numbers) / len(numbers)
