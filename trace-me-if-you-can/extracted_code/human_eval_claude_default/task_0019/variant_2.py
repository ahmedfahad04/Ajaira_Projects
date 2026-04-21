def sort_numbers(numbers):
    ordering = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    word_list = list(filter(None, numbers.split(' ')))
    return ' '.join(sorted(word_list, key=ordering.index))
