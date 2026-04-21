def sort_and_interleave(lst):
    even_elements = sorted(lst[::2])
    odd_elements = lst[1::2]
    interleaved = []
    for even, odd in zip(even_elements, odd_elements):
        interleaved.extend([even, odd])
    if len(even_elements) > len(odd_elements):
        interleaved.append(even_elements[-1])
    return interleaved
