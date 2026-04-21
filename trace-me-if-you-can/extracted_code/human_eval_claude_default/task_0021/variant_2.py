# Variant 2: Functional approach with map and lambda
def normalize(numbers):
    if not numbers:
        return []
    
    extremes = (min(numbers), max(numbers))
    denominator = extremes[1] - extremes[0]
    
    if denominator == 0:
        return [0.0] * len(numbers)
    
    return list(map(lambda x: (x - extremes[0]) / denominator, numbers))
