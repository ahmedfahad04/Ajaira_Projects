# Version 5: Generator-based approach with alternating pattern
def interleave_sorted():
    sorted_evens = iter(sorted(l[::2]))
    odds_iter = iter(l[1::2])
    
    for _ in range(len(l)):
        try:
            yield next(sorted_evens)
            if len(l) > 1:  # Check if there are odd elements
                yield next(odds_iter)
        except StopIteration:
            break

return list(interleave_sorted())
