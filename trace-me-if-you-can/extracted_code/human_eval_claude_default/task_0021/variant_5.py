# Variant 5: Reduce-based approach with tuple unpacking
from functools import reduce

def normalize(numbers):
    if not numbers:
        return []
    
    # Find min and max using reduce
    min_max = reduce(
        lambda acc, x: (min(acc[0], x), max(acc[1], x)),
        numbers[1:],
        (numbers[0], numbers[0])
    )
    
    min_num, max_num = min_max
    divisor = max_num - min_num
    
    return [0.0] * len(numbers) if divisor == 0 else [(x - min_num) / divisor for x in numbers]
