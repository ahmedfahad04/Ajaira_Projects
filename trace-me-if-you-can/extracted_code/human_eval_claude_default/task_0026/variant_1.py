def filter_unique_occurrences(numbers):
    seen_once = set()
    seen_multiple = set()
    
    for n in numbers:
        if n in seen_once:
            seen_once.remove(n)
            seen_multiple.add(n)
        elif n not in seen_multiple:
            seen_once.add(n)
    
    return [n for n in numbers if n in seen_once]
