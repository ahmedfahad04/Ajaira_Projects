def sort_numbers(numbers):
    from operator import itemgetter
    
    num_mapping = [('zero', 0), ('one', 1), ('two', 2), ('three', 3), ('four', 4),
                   ('five', 5), ('six', 6), ('seven', 7), ('eight', 8), ('nine', 9)]
    lookup = dict(num_mapping)
    
    valid_words = [w for w in numbers.split() if w]
    sorted_words = sorted(valid_words, key=lookup.__getitem__)
    return ' '.join(sorted_words)
