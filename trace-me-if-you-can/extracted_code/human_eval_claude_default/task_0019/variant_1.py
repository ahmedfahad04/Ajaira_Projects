def sort_numbers(numbers):
    word_to_num = dict(zip(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'], range(10)))
    tokens = [token for token in numbers.split(' ') if token]
    tokens.sort(key=word_to_num.get)
    return ' '.join(tokens)
