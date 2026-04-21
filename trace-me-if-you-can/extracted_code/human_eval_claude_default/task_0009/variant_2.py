def running_maximum(numbers):
    def max_generator():
        current_max = float('-inf')
        for n in numbers:
            current_max = n if current_max == float('-inf') else max(current_max, n)
            yield current_max
    
    return list(max_generator())
