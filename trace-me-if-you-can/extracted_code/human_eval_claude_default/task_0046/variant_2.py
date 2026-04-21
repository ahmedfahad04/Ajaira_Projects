# Version 2: Generator-based approach with itertools
from itertools import islice

def sequence_generator():
    a, b, c, d = 0, 0, 2, 0
    yield a
    yield b
    yield c
    yield d
    
    while True:
        next_val = a + b + c + d
        yield next_val
        a, b, c, d = b, c, d, next_val

return next(islice(sequence_generator(), n, n + 1))
