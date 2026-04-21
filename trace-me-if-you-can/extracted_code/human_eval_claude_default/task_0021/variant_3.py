# Variant 3: Generator-based approach with deferred computation
def normalize(numbers):
    if not numbers:
        return []
    
    # Pre-compute bounds
    bounds = {'min': min(numbers), 'max': max(numbers)}
    scale = bounds['max'] - bounds['min']
    
    if scale == 0:
        return [0.0] * len(numbers)
    
    def generate_normalized():
        for value in numbers:
            yield (value - bounds['min']) / scale
    
    return list(generate_normalized())
