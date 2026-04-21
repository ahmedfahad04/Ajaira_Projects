from itertools import accumulate

def running_maximum(numbers):
    return list(accumulate(numbers, max))
